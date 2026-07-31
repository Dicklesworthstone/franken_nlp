//! Release dependency-policy enforcement.
//!
//! The test deliberately invokes Cargo at test time rather than adding a TOML
//! parser or graph library to the release dependency graph.  It only evaluates
//! normal/build dependency edges; dev dependencies are inventoried separately
//! and do not participate in the release-graph assertions.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::Path;
use std::process::Command;

use serde::Deserialize;

const COMMODITY_ROOTS: &[&str] = &["clap", "serde", "serde_json", "sha2"];
const SUITE_ROOTS: &[&str] = &[
    "asupersync",
    "ft-core",
    "ft-kernel-cpu",
    "ft-serialize",
    "fsqlite",
    "fsqlite-types",
];
const FORBIDDEN_DIRECT: &[&str] = &[
    "anyhow",
    "ctrlc",
    "half",
    "hyper",
    "llguidance",
    "memmap2",
    "minijinja",
    "num_cpus",
    "rayon",
    "reqwest",
    "rusqlite",
    "thiserror",
    "tiktoken",
    "tokenizers",
    "uuid",
];
const FORBIDDEN_RELEASE_GRAPH: &[&str] = &["rayon", "rayon-core"];
const SUPPLY_CHAIN_MANIFEST: &str = "docs/SUPPLY_CHAIN_MANIFEST.json";

#[derive(Debug, Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    resolve: Resolve,
}

#[derive(Debug, Deserialize)]
struct Package {
    id: String,
    name: String,
    version: String,
    #[serde(default)]
    dependencies: Vec<ManifestDependency>,
}

#[derive(Debug, Deserialize)]
struct ManifestDependency {
    name: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    optional: bool,
    #[serde(default)]
    source: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Resolve {
    root: Option<String>,
    #[serde(default)]
    nodes: Vec<ResolveNode>,
}

#[derive(Debug, Deserialize)]
struct ResolveNode {
    id: String,
    #[serde(default)]
    deps: Vec<ResolveDependency>,
}

#[derive(Debug, Deserialize)]
struct ResolveDependency {
    pkg: String,
    #[serde(default)]
    dep_kinds: Vec<DependencyKind>,
}

#[derive(Debug, Deserialize)]
struct DependencyKind {
    #[serde(default)]
    kind: Option<String>,
}

#[derive(Debug)]
struct PolicyFailure {
    offending_crate: String,
    full_path: String,
    detail: String,
}

#[derive(Debug)]
struct PolicyReport {
    release_direct_roots: BTreeSet<String>,
    dev_direct_roots: BTreeSet<String>,
    release_packages: BTreeSet<String>,
    release_package_inventory: BTreeSet<String>,
}

#[derive(Debug, Deserialize)]
struct SupplyChainManifest {
    schema_version: u32,
    scope: String,
    direct_release_roots: Vec<String>,
    packages: Vec<SupplyChainPackage>,
}

#[derive(Debug, Deserialize)]
struct SupplyChainPackage {
    name: String,
    version: String,
}

#[test]
fn current_release_graph_obeys_dependency_policy() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    match current_policy(root) {
        Ok(report) => report_pass(&report),
        Err(failure) => report_failure(&failure),
    }
}

#[test]
fn synthetic_metadata_rejects_new_direct_roots_and_names_the_full_path() {
    let metadata = parse_fixture(
        r#"
        {
          "packages": [
            {
              "id": "franken_nlp 0.1.0 (path+file:///fixture)",
              "name": "franken_nlp",
              "version": "0.1.0",
              "dependencies": [
                {"name": "clap", "kind": null},
                {"name": "serde", "kind": null},
                {"name": "serde_json", "kind": null},
                {"name": "sha2", "kind": null},
                {"name": "reqwest", "kind": null}
              ]
            },
            {"id": "reqwest 1.0.0 (registry+fixture)", "name": "reqwest", "version": "1.0.0"}
          ],
          "resolve": {
            "root": "franken_nlp 0.1.0 (path+file:///fixture)",
            "nodes": [
              {
                "id": "franken_nlp 0.1.0 (path+file:///fixture)",
                "deps": [
                  {"pkg": "reqwest 1.0.0 (registry+fixture)", "dep_kinds": [{"kind": null}]}
                ]
              },
              {"id": "reqwest 1.0.0 (registry+fixture)", "deps": []}
            ]
          }
        }
        "#,
    );
    assert!(
        metadata.is_ok(),
        "synthetic direct-root metadata must parse: {metadata:?}"
    );
    let Some(metadata) = metadata.ok() else {
        return;
    };

    let result = evaluate_policy(
        &metadata,
        "franken_nlp\n└── reqwest v1.0.0",
        &BTreeSet::new(),
    );
    assert!(result.is_err(), "a fourth release root must fail");
    let Some(failure) = result.err() else {
        return;
    };
    assert_eq!(failure.offending_crate, "reqwest");
    assert_eq!(failure.full_path, "franken_nlp@0.1.0 -> reqwest@1.0.0");
}

#[test]
fn synthetic_metadata_rejects_transitive_rayon_and_names_the_full_path() {
    let metadata = parse_fixture(
        r#"
        {
          "packages": [
            {
              "id": "franken_nlp 0.1.0 (path+file:///fixture)",
              "name": "franken_nlp",
              "version": "0.1.0",
              "dependencies": [
                {"name": "clap", "kind": null},
                {"name": "serde", "kind": null},
                {"name": "serde_json", "kind": null},
                {"name": "sha2", "kind": null},
                {"name": "ft-core", "kind": null, "source": "git+https://example.invalid/frankentorch?rev=0123456789012345678901234567890123456789#0123456789012345678901234567890123456789"}
              ]
            },
            {"id": "ft-core 0.1.0 (git+fixture)", "name": "ft-core", "version": "0.1.0"},
            {"id": "rayon 1.0.0 (registry+fixture)", "name": "rayon", "version": "1.0.0"}
          ],
          "resolve": {
            "root": "franken_nlp 0.1.0 (path+file:///fixture)",
            "nodes": [
              {
                "id": "franken_nlp 0.1.0 (path+file:///fixture)",
                "deps": [
                  {"pkg": "ft-core 0.1.0 (git+fixture)", "dep_kinds": [{"kind": null}]}
                ]
              },
              {
                "id": "ft-core 0.1.0 (git+fixture)",
                "deps": [
                  {"pkg": "rayon 1.0.0 (registry+fixture)", "dep_kinds": [{"kind": null}]}
                ]
              },
              {"id": "rayon 1.0.0 (registry+fixture)", "deps": []}
            ]
          }
        }
        "#,
    );
    assert!(
        metadata.is_ok(),
        "synthetic Rayon metadata must parse: {metadata:?}"
    );
    let Some(metadata) = metadata.ok() else {
        return;
    };

    let result = evaluate_policy(
        &metadata,
        "franken_nlp\n└── ft-core v0.1.0\n    └── rayon v1.0.0",
        &BTreeSet::new(),
    );
    assert!(
        result.is_err(),
        "transitive Rayon must fail the release graph"
    );
    let Some(failure) = result.err() else {
        return;
    };
    assert_eq!(failure.offending_crate, "rayon");
    assert_eq!(
        failure.full_path,
        "franken_nlp@0.1.0 -> ft-core@0.1.0 -> rayon@1.0.0"
    );
}

#[test]
fn synthetic_metadata_excludes_dev_only_dependencies_from_release_graph() {
    let metadata = parse_fixture(
        r#"
        {
          "packages": [
            {
              "id": "franken_nlp 0.1.0 (path+file:///fixture)",
              "name": "franken_nlp",
              "version": "0.1.0",
              "dependencies": [
                {"name": "clap", "kind": null},
                {"name": "serde", "kind": null},
                {"name": "serde_json", "kind": null},
                {"name": "sha2", "kind": null},
                {"name": "rayon", "kind": "dev"}
              ]
            },
            {"id": "rayon 1.0.0 (registry+fixture)", "name": "rayon", "version": "1.0.0"}
          ],
          "resolve": {
            "root": "franken_nlp 0.1.0 (path+file:///fixture)",
            "nodes": [
              {
                "id": "franken_nlp 0.1.0 (path+file:///fixture)",
                "deps": [
                  {"pkg": "rayon 1.0.0 (registry+fixture)", "dep_kinds": [{"kind": "dev"}]}
                ]
              },
              {"id": "rayon 1.0.0 (registry+fixture)", "deps": []}
            ]
          }
        }
        "#,
    );
    assert!(
        metadata.is_ok(),
        "synthetic dev-only metadata must parse: {metadata:?}"
    );
    let Some(metadata) = metadata.ok() else {
        return;
    };

    let result = evaluate_policy(&metadata, "franken_nlp", &BTreeSet::new());
    assert!(
        result.is_ok(),
        "a dev-only dependency must not enter the release graph: {result:?}"
    );
    let Some(report) = result.ok() else {
        return;
    };
    assert_eq!(
        report.dev_direct_roots,
        BTreeSet::from(["rayon".to_owned()])
    );
    assert!(!report.release_packages.contains("rayon"));
}

fn current_policy(root: &Path) -> Result<PolicyReport, PolicyFailure> {
    let metadata = cargo_metadata(root)?;
    let cargo_tree = cargo_tree(root)?;
    let lock_packages = cargo_lock_packages(root)?;
    let report = evaluate_policy(&metadata, &cargo_tree, &lock_packages)?;
    validate_supply_chain_manifest(root, &report)?;
    Ok(report)
}

fn cargo_metadata(root: &Path) -> Result<Metadata, PolicyFailure> {
    let output = cargo_command(root, ["metadata", "--locked", "--format-version", "1"])?;
    serde_json::from_str(&output).map_err(|error| PolicyFailure {
        offending_crate: "cargo-metadata".to_owned(),
        full_path: root.display().to_string(),
        detail: format!("cannot parse cargo metadata JSON: {error}"),
    })
}

fn cargo_tree(root: &Path) -> Result<String, PolicyFailure> {
    cargo_command(root, ["tree", "--locked", "--edges", "normal,build"])
}

fn cargo_command<const N: usize>(
    root: &Path,
    arguments: [&str; N],
) -> Result<String, PolicyFailure> {
    let command = format!("cargo {}", arguments.join(" "));
    let output = Command::new(env!("CARGO"))
        .current_dir(root)
        .args(arguments)
        .output()
        .map_err(|error| PolicyFailure {
            offending_crate: "cargo".to_owned(),
            full_path: command.clone(),
            detail: format!("cannot invoke Cargo: {error}"),
        })?;
    if !output.status.success() {
        return Err(PolicyFailure {
            offending_crate: "cargo".to_owned(),
            full_path: command,
            detail: format!(
                "Cargo command failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            ),
        });
    }
    String::from_utf8(output.stdout).map_err(|error| PolicyFailure {
        offending_crate: "cargo".to_owned(),
        full_path: command,
        detail: format!("Cargo emitted non-UTF-8 dependency data: {error}"),
    })
}

fn cargo_lock_packages(root: &Path) -> Result<BTreeSet<String>, PolicyFailure> {
    let lock_path = root.join("Cargo.lock");
    let lock = fs::read_to_string(&lock_path).map_err(|error| PolicyFailure {
        offending_crate: "Cargo.lock".to_owned(),
        full_path: lock_path.display().to_string(),
        detail: format!("cannot read lockfile: {error}"),
    })?;
    let packages: BTreeSet<String> = lock
        .split("[[package]]")
        .skip(1)
        .filter_map(|section| {
            section.lines().map(str::trim).find_map(|line| {
                line.strip_prefix("name = \"")
                    .and_then(|value| value.strip_suffix('\"'))
            })
        })
        .map(ToOwned::to_owned)
        .collect();
    if packages.is_empty() {
        return Err(PolicyFailure {
            offending_crate: "Cargo.lock".to_owned(),
            full_path: lock_path.display().to_string(),
            detail: "lockfile contains no [[package]] entries".to_owned(),
        });
    }
    Ok(packages)
}

fn evaluate_policy(
    metadata: &Metadata,
    cargo_tree: &str,
    lock_packages: &BTreeSet<String>,
) -> Result<PolicyReport, PolicyFailure> {
    let root = root_package(metadata)?;
    let root_label = package_label(root);
    let mut release_direct_roots = BTreeSet::new();
    let mut dev_direct_roots = BTreeSet::new();

    for dependency in &root.dependencies {
        if dependency.kind.as_deref() == Some("dev") {
            dev_direct_roots.insert(dependency.name.clone());
            continue;
        }
        if dependency.optional {
            validate_direct_dependency(dependency, &root_label)?;
            continue;
        }
        release_direct_roots.insert(dependency.name.clone());
        validate_direct_dependency(dependency, &root_label)?;
    }

    for required in COMMODITY_ROOTS {
        if !release_direct_roots.contains(*required) {
            return Err(PolicyFailure {
                offending_crate: (*required).to_owned(),
                full_path: root_label.clone(),
                detail: "required commodity dependency family is missing from the release roots"
                    .to_owned(),
            });
        }
    }

    let release_ids = release_package_ids(metadata, cargo_tree)?;
    for package_id in &release_ids {
        let package = package_by_id(metadata, package_id).ok_or_else(|| PolicyFailure {
            offending_crate: package_id.clone(),
            full_path: root_label.clone(),
            detail: "cargo metadata resolve graph names an unknown package id".to_owned(),
        })?;
        if !lock_packages.is_empty() && !lock_packages.contains(&package.name) {
            return Err(PolicyFailure {
                offending_crate: package.name.clone(),
                full_path: full_path_to_id(metadata, cargo_tree, package_id)
                    .unwrap_or_else(|| root_label.clone()),
                detail: "release package is absent from Cargo.lock".to_owned(),
            });
        }
    }

    for forbidden in FORBIDDEN_RELEASE_GRAPH {
        if cargo_tree_mentions(cargo_tree, forbidden) {
            return Err(PolicyFailure {
                offending_crate: (*forbidden).to_owned(),
                full_path: full_path_to_named_package(metadata, cargo_tree, forbidden)
                    .unwrap_or_else(|| format!("{root_label} -> {forbidden}")),
                detail: "cargo tree found a forbidden crate in the release graph".to_owned(),
            });
        }
        if let Some(path) = full_path_to_named_package(metadata, cargo_tree, forbidden) {
            return Err(PolicyFailure {
                offending_crate: (*forbidden).to_owned(),
                full_path: path,
                detail: "cargo metadata found a forbidden crate in the release graph".to_owned(),
            });
        }
    }

    let release_packages = release_ids
        .iter()
        .filter_map(|id| package_by_id(metadata, id).map(|package| package.name.clone()))
        .collect();
    let release_package_inventory = release_ids
        .iter()
        .filter_map(|id| package_by_id(metadata, id).map(package_inventory_label))
        .collect();
    Ok(PolicyReport {
        release_direct_roots,
        dev_direct_roots,
        release_packages,
        release_package_inventory,
    })
}

fn validate_supply_chain_manifest(root: &Path, report: &PolicyReport) -> Result<(), PolicyFailure> {
    let path = root.join(SUPPLY_CHAIN_MANIFEST);
    let document = fs::read_to_string(&path).map_err(|error| PolicyFailure {
        offending_crate: "supply-chain-manifest".to_owned(),
        full_path: path.display().to_string(),
        detail: format!("cannot read committed supply-chain manifest: {error}"),
    })?;
    let manifest: SupplyChainManifest =
        serde_json::from_str(&document).map_err(|error| PolicyFailure {
            offending_crate: "supply-chain-manifest".to_owned(),
            full_path: path.display().to_string(),
            detail: format!("cannot parse supply-chain manifest JSON: {error}"),
        })?;
    if manifest.schema_version != 1 || manifest.scope != "release-normal-build" {
        return Err(PolicyFailure {
            offending_crate: "supply-chain-manifest".to_owned(),
            full_path: path.display().to_string(),
            detail: "manifest schema_version/scope is not the release normal/build contract"
                .to_owned(),
        });
    }
    let manifest_roots: BTreeSet<String> = manifest.direct_release_roots.into_iter().collect();
    if manifest_roots != report.release_direct_roots {
        return Err(PolicyFailure {
            offending_crate: "supply-chain-manifest".to_owned(),
            full_path: path.display().to_string(),
            detail: format!(
                "direct-root drift expected={:?} observed={manifest_roots:?}",
                report.release_direct_roots
            ),
        });
    }
    let manifest_inventory: BTreeSet<String> = manifest
        .packages
        .into_iter()
        .map(|package| format!("{}@{}", package.name, package.version))
        .collect();
    if manifest_inventory.len() != report.release_package_inventory.len()
        || manifest_inventory != report.release_package_inventory
    {
        return Err(PolicyFailure {
            offending_crate: "supply-chain-manifest".to_owned(),
            full_path: path.display().to_string(),
            detail: format!(
                "release package inventory drift missing={:?} unexpected={:?}",
                report
                    .release_package_inventory
                    .difference(&manifest_inventory)
                    .collect::<Vec<_>>(),
                manifest_inventory
                    .difference(&report.release_package_inventory)
                    .collect::<Vec<_>>(),
            ),
        });
    }
    Ok(())
}

fn validate_direct_dependency(
    dependency: &ManifestDependency,
    root_label: &str,
) -> Result<(), PolicyFailure> {
    let name = dependency.name.as_str();
    if FORBIDDEN_DIRECT.contains(&name) || is_allocator_crate(name) {
        return Err(PolicyFailure {
            offending_crate: dependency.name.clone(),
            full_path: format!("{root_label} -> {name}"),
            detail: "named forbidden crate or allocator crate is a direct release dependency"
                .to_owned(),
        });
    }
    if !COMMODITY_ROOTS.contains(&name) && !SUITE_ROOTS.contains(&name) {
        return Err(PolicyFailure {
            offending_crate: dependency.name.clone(),
            full_path: format!("{root_label} -> {name}"),
            detail: "direct release dependency is outside the three commodity families and pinned FrankenSuite".to_owned(),
        });
    }
    if SUITE_ROOTS.contains(&name) && !is_immutable_git_pin(dependency.source.as_deref()) {
        return Err(PolicyFailure {
            offending_crate: dependency.name.clone(),
            full_path: format!("{root_label} -> {name}"),
            detail: "FrankenSuite dependency is not pinned to an immutable Git revision".to_owned(),
        });
    }
    Ok(())
}

fn is_allocator_crate(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    normalized.contains("alloc")
        || matches!(
            normalized.as_str(),
            "dlmalloc" | "jemallocator" | "mimalloc" | "snmalloc-rs" | "talc"
        )
}

fn is_immutable_git_pin(source: Option<&str>) -> bool {
    let Some(source) = source else {
        return false;
    };
    let Some((_, revision)) = source.rsplit_once('#') else {
        return false;
    };
    source.starts_with("git+")
        && revision.len() == 40
        && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn root_package(metadata: &Metadata) -> Result<&Package, PolicyFailure> {
    let root_id = metadata
        .resolve
        .root
        .as_deref()
        .ok_or_else(|| PolicyFailure {
            offending_crate: "cargo-metadata".to_owned(),
            full_path: "<missing-root>".to_owned(),
            detail: "cargo metadata did not identify the package root".to_owned(),
        })?;
    package_by_id(metadata, root_id).ok_or_else(|| PolicyFailure {
        offending_crate: "cargo-metadata".to_owned(),
        full_path: root_id.to_owned(),
        detail: "cargo metadata root id is absent from packages".to_owned(),
    })
}

fn release_package_ids(
    metadata: &Metadata,
    cargo_tree: &str,
) -> Result<BTreeSet<String>, PolicyFailure> {
    let root_id = metadata
        .resolve
        .root
        .as_deref()
        .ok_or_else(|| PolicyFailure {
            offending_crate: "cargo-metadata".to_owned(),
            full_path: "<missing-root>".to_owned(),
            detail: "cargo metadata did not identify the package root".to_owned(),
        })?;
    let nodes: BTreeMap<&str, &ResolveNode> = metadata
        .resolve
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect();
    let mut reachable = BTreeSet::new();
    let mut queue = VecDeque::from([root_id.to_owned()]);
    while let Some(id) = queue.pop_front() {
        if !reachable.insert(id.clone()) {
            continue;
        }
        let node = nodes.get(id.as_str()).ok_or_else(|| PolicyFailure {
            offending_crate: "cargo-metadata".to_owned(),
            full_path: id.clone(),
            detail: "release graph references a package without a resolve node".to_owned(),
        })?;
        for dependency in &node.deps {
            if is_active_release_edge(metadata, cargo_tree, dependency) {
                queue.push_back(dependency.pkg.clone());
            }
        }
    }
    Ok(reachable)
}

fn is_release_edge(dependency: &ResolveDependency) -> bool {
    dependency.dep_kinds.is_empty()
        || dependency
            .dep_kinds
            .iter()
            .any(|kind| kind.kind.as_deref() != Some("dev"))
}

fn is_active_release_edge(
    metadata: &Metadata,
    cargo_tree: &str,
    dependency: &ResolveDependency,
) -> bool {
    is_release_edge(dependency)
        && package_by_id(metadata, &dependency.pkg)
            .is_some_and(|package| cargo_tree_mentions(cargo_tree, &package.name))
}

fn package_by_id<'a>(metadata: &'a Metadata, id: &str) -> Option<&'a Package> {
    metadata.packages.iter().find(|package| package.id == id)
}

fn full_path_to_named_package(
    metadata: &Metadata,
    cargo_tree: &str,
    name: &str,
) -> Option<String> {
    let target = release_package_ids(metadata, cargo_tree)
        .ok()?
        .into_iter()
        .find(|id| package_by_id(metadata, id).is_some_and(|package| package.name == name))?;
    full_path_to_id(metadata, cargo_tree, &target)
}

fn full_path_to_id(metadata: &Metadata, cargo_tree: &str, target: &str) -> Option<String> {
    let root = metadata.resolve.root.as_deref()?;
    let nodes: BTreeMap<&str, &ResolveNode> = metadata
        .resolve
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect();
    let mut previous: BTreeMap<String, String> = BTreeMap::new();
    let mut seen = BTreeSet::from([root.to_owned()]);
    let mut queue = VecDeque::from([root.to_owned()]);

    while let Some(id) = queue.pop_front() {
        if id == target {
            let mut ids = vec![id];
            while let Some(parent) = previous.get(ids.last()?) {
                ids.push(parent.clone());
            }
            ids.reverse();
            return ids
                .iter()
                .map(|id| package_by_id(metadata, id).map(package_label))
                .collect::<Option<Vec<_>>>()
                .map(|labels| labels.join(" -> "));
        }
        let node = nodes.get(id.as_str())?;
        for dependency in &node.deps {
            if is_active_release_edge(metadata, cargo_tree, dependency)
                && seen.insert(dependency.pkg.clone())
            {
                previous.insert(dependency.pkg.clone(), id.clone());
                queue.push_back(dependency.pkg.clone());
            }
        }
    }
    None
}

fn package_label(package: &Package) -> String {
    format!("{}@{}", package.name, package.version)
}

fn package_inventory_label(package: &Package) -> String {
    package_label(package)
}

fn cargo_tree_mentions(cargo_tree: &str, package: &str) -> bool {
    cargo_tree.lines().any(|line| {
        line.split_whitespace().any(|word| {
            word.trim_matches(|character: char| matches!(character, '├' | '└' | '│' | '─'))
                == package
        })
    })
}

fn parse_fixture(document: &str) -> Result<Metadata, String> {
    serde_json::from_str(document).map_err(|error| error.to_string())
}

fn report_pass(report: &PolicyReport) {
    eprintln!(
        "DEP_POLICY direct_release_roots={}",
        join_inventory(&report.release_direct_roots)
    );
    eprintln!(
        "DEP_POLICY direct_dev_roots={}",
        join_inventory(&report.dev_direct_roots)
    );
    eprintln!(
        "DEP_POLICY release_packages={}",
        join_inventory(&report.release_packages)
    );
    eprintln!("DEP_POLICY rayon_trace=none");
    eprintln!("DEP_POLICY RESULT=PASS violation=none");
}

fn report_failure(failure: &PolicyFailure) {
    eprintln!("DEP_POLICY violation={}", failure.offending_crate);
    eprintln!("DEP_POLICY path={}", failure.full_path);
    eprintln!("DEP_POLICY detail={}", failure.detail);
    eprintln!(
        "DEP_POLICY RESULT=FAIL violation={}",
        failure.offending_crate
    );
    assert!(
        false,
        "dependency policy violation: {} ({})",
        failure.offending_crate, failure.full_path
    );
}

fn join_inventory(entries: &BTreeSet<String>) -> String {
    if entries.is_empty() {
        "none".to_owned()
    } else {
        entries
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(",")
    }
}
