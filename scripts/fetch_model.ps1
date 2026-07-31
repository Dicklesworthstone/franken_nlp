# Download the pinned Nanbeige4.2-3B conversion source closure. This is a
# human-run provisioning helper, never an installer or an inference path.
[CmdletBinding()]
param(
    [string]$Dest,
    [string]$Revision = 'f56ec5a9650268aa098496734743c25ea778bd2d',
    [switch]$AllowUntrustedRevision,
    [string]$Catalog,
    [switch]$CheckOnly,
    [string]$TestBaseUrl
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$DefaultRevision = 'f56ec5a9650268aa098496734743c25ea778bd2d'
$Model = 'Nanbeige4.2-3B'
$DefaultBase = 'https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve'
$MarginBytes = [int64]67108864
$script:LockDir = $null
$script:JournalDir = $null

function Write-Log([string]$Message) {
    [Console]::Error.WriteLine("{0} FETCH_MODEL {1}" -f [DateTime]::UtcNow.ToString('o'), $Message)
}

function Write-Usage {
    [Console]::Error.WriteLine('Usage: scripts/fetch_model.ps1 [-Dest DIR] [-CheckOnly]')
    [Console]::Error.WriteLine('       scripts/fetch_model.ps1 -Revision REV -AllowUntrustedRevision -Catalog FILE [-Dest DIR]')
    [Console]::Error.WriteLine('Downloads the pinned source closure for fnlp convert; it is not an end-user installer.')
}

function Write-ResumeGuidance {
    [Console]::Error.WriteLine(('RESUME command=pwsh -File scripts/fetch_model.ps1 -Dest "{0}" journal_dir="{1}" note="retry resumes secure journal-bound partials rather than restarting"' -f $Dest, $script:JournalDir))
}

function Release-Lock {
    if ($null -ne $script:LockDir -and (Test-Path -LiteralPath $script:LockDir -PathType Container)) {
        try { Remove-Item -LiteralPath $script:LockDir -Force -ErrorAction Stop }
        catch { Write-Log "LOCK_RELEASE_DEFERRED path=$script:LockDir" }
    }
}

function Exit-Fetch([int]$Code, [string]$Message = '') {
    if ($Message) { Write-Log "ERROR code=$Code detail=$Message" }
    if ($Code -ne 0) { Write-ResumeGuidance }
    Release-Lock
    exit $Code
}

function Get-DefaultCatalog {
@'
model-00001-of-00002.safetensors|4973547960|09d265d5ec837bc64462796b7f8c110be9a135a55ed7a6eb5d07e0e90c976a94
model-00002-of-00002.safetensors|3366076760|31019e7870a044f44bc3f7e981f8c5ecd42d341e5ca6cfdbfd07fb95d95be389
model.safetensors.index.json|16519|30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1
config.json|1019|f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19
tokenizer.json|18450979|1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f
tokenizer.model|2782298|fb41d04798b714520a9b075727b0226538b7330254299062742c50ec8374bc36
tokenizer_config.json|10990|3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518
added_tokens.json|174|9e3b127a27647df2c353cc1e5500826f7cdbe8bd15a458e368bba8422e9719cf
special_tokens_map.json|623|b718fce2b7a8940ffeddc1e67f3b092cc0d13ac885c63a021528786f8c4cf6c0
generation_config.json|187|68c690ce23efb6caae30c006ff3c1efd826297ff1df4338c04f7ac6f685d8746
'@ -split "`r?`n" | Where-Object { $_ }
}

function Get-Catalog {
    if ($Catalog) {
        if (-not (Test-Path -LiteralPath $Catalog -PathType Leaf)) { Exit-Fetch 2 "catalog does not exist: $Catalog" }
        $lines = Get-Content -LiteralPath $Catalog
    } else { $lines = Get-DefaultCatalog }
    $seen = @{}
    $entries = foreach ($line in $lines) {
        if (-not $line.Trim()) { continue }
        $parts = $line -split '\|', 3
        if ($parts.Count -ne 3 -or $parts[0] -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or $parts[0] -match '\.\.' -or $parts[1] -notmatch '^\d+$' -or $parts[2] -notmatch '^[a-f0-9]{64}$' -or $seen.ContainsKey($parts[0])) {
            Exit-Fetch 2 "malformed catalog entry: $line"
        }
        $seen[$parts[0]] = $true
        [PSCustomObject]@{ Name = $parts[0]; Length = [int64]$parts[1]; Sha256 = $parts[2] }
    }
    if (-not $entries -or @($entries).Count -eq 0) { Exit-Fetch 2 'catalog has no entries' }
    if (-not $Catalog) {
        $total = (@($entries) | Measure-Object -Property Length -Sum).Sum
        if (@($entries).Count -ne 10 -or $total -ne 8360887509) { Exit-Fetch 1 'embedded catalog invariant failed' }
    }
    return @($entries)
}

function Test-ReparsePoint([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    return (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
}

function Assert-OwnerControlledDirectory([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer -or (Test-ReparsePoint $Path)) { Exit-Fetch 4 "destination must be a non-reparse directory: $Path" }
    if ($env:OS -eq 'Windows_NT') {
        $owner = (Get-Acl -LiteralPath $Path).Owner
        $current = [Security.Principal.WindowsIdentity]::GetCurrent().Name
        if ($owner -ne $current) { Exit-Fetch 4 "destination is not owned by the current user: $Path" }
    }
}

function Test-SecureRegularFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if (Test-ReparsePoint $Path) { return $false }
    if ($env:OS -eq 'Windows_NT') {
        $owner = (Get-Acl -LiteralPath $Path).Owner
        if ($owner -ne [Security.Principal.WindowsIdentity]::GetCurrent().Name) { return $false }
    }
    return $true
}

function Ensure-Destination {
    New-Item -ItemType Directory -LiteralPath $Dest -Force | Out-Null
    Assert-OwnerControlledDirectory $Dest
    $script:JournalDir = Join-Path $Dest '.fnlp-fetch-journals'
    New-Item -ItemType Directory -LiteralPath $script:JournalDir -Force | Out-Null
    Assert-OwnerControlledDirectory $script:JournalDir
}

function Get-Sha256([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Get-Length([string]$Path) { [int64](Get-Item -LiteralPath $Path -Force).Length }

function Test-VerifiedFile([string]$Path, [int64]$ExpectedLength, [string]$ExpectedSha256) {
    if (-not (Test-SecureRegularFile $Path)) {
        $kind = if (Test-Path -LiteralPath $Path) { 'reparse-or-owner-refused' } else { 'missing' }
        Write-Log "VERIFY RESULT=FAIL file=$([IO.Path]::GetFileName($Path)) expected_bytes=$ExpectedLength observed_bytes=$kind expected_sha256=$ExpectedSha256 observed_sha256=$kind"
        return $false
    }
    $actualLength = Get-Length $Path
    $actualSha256 = Get-Sha256 $Path
    if ($actualLength -ne $ExpectedLength -or $actualSha256 -ne $ExpectedSha256) {
        Write-Log "VERIFY RESULT=FAIL file=$([IO.Path]::GetFileName($Path)) expected_bytes=$ExpectedLength observed_bytes=$actualLength expected_sha256=$ExpectedSha256 observed_sha256=$actualSha256"
        return $false
    }
    Write-Log "VERIFY RESULT=PASS file=$([IO.Path]::GetFileName($Path)) observed_bytes=$actualLength sha256=$actualSha256"
    return $true
}

function Quarantine([string]$Path, [string]$Reason) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $qdir = Join-Path $Dest 'quarantine'
    New-Item -ItemType Directory -LiteralPath $qdir -Force | Out-Null
    Assert-OwnerControlledDirectory $qdir
    $observed = if (Test-SecureRegularFile $Path) { Get-Sha256 $Path } elseif (Test-ReparsePoint $Path) { 'reparse-refused' } else { 'unreadable' }
    $target = Join-Path $qdir (([IO.Path]::GetFileName($Path)) + '.' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '.' + $PID + '.' + $observed)
    Move-Item -LiteralPath $Path -Destination $target -ErrorAction Stop
    Add-Content -LiteralPath (Join-Path $qdir 'quarantine.log') -Value ("{0} QUARANTINE file={1} target={2} observed_sha256={3} reason={4}" -f [DateTime]::UtcNow.ToString('o'), $Path, $target, $observed, $Reason)
    Write-Log "QUARANTINE file=$([IO.Path]::GetFileName($Path)) observed_sha256=$observed reason=$Reason"
}

function Get-JournalText([string]$Url, $Entry) {
    "url=$Url`nrevision=$Revision`nname=$($Entry.Name)`nbytes=$($Entry.Length)`nsha256=$($Entry.Sha256)`n"
}

function Write-NewUtf8File([string]$Path, [string]$Content) {
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false), 4096, $true)
        try { $writer.Write($Content); $writer.Flush() } finally { $writer.Dispose() }
        $stream.Flush($true)
    } finally { $stream.Dispose() }
}

function Test-Journal([string]$Path, [string]$Url, $Entry) {
    if (-not (Test-SecureRegularFile $Path)) { return $false }
    return ((Get-Content -LiteralPath $Path -Raw) -eq (Get-JournalText $Url $Entry))
}

function Ensure-PartialAndJournal([string]$Partial, [string]$Journal, [string]$Url, $Entry) {
    if (Test-Path -LiteralPath $Partial -PathType Leaf) {
        if (Test-ReparsePoint $Partial) { Exit-Fetch 1 "refused reparse-point partial=$Partial" }
        if ((Test-Path -LiteralPath $Journal) -and (Test-ReparsePoint $Journal)) { Exit-Fetch 1 "refused reparse-point journal=$Journal" }
        if (-not (Test-SecureRegularFile $Partial) -or -not (Test-Journal $Journal $Url $Entry)) {
            Quarantine $Partial 'unbound-or-mismatched-partial'
            if (Test-Path -LiteralPath $Journal) { Quarantine $Journal 'mismatched-journal' }
        }
    } elseif (Test-Path -LiteralPath $Journal) { Quarantine $Journal 'orphaned-journal' }
    if (-not (Test-Path -LiteralPath $Partial)) {
        try { [IO.File]::Open($Partial, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None).Dispose() }
        catch { Exit-Fetch 1 "refused to create secure partial=$Partial" }
    }
    if (-not (Test-Path -LiteralPath $Journal)) {
        try { Write-NewUtf8File $Journal (Get-JournalText $Url $Entry) }
        catch { Exit-Fetch 1 "refused to create journal=$Journal" }
    }
    if (-not (Test-SecureRegularFile $Partial) -or -not (Test-Journal $Journal $Url $Entry)) { Exit-Fetch 1 "partial/journal binding rejected name=$($Entry.Name)" }
}

function Assert-Preflight($Entries) {
    $total = [int64](($Entries | Measure-Object -Property Length -Sum).Sum)
    $largest = [int64](($Entries | Measure-Object -Property Length -Maximum).Maximum)
    $free = [int64](Get-Item -LiteralPath $Dest).PSDrive.Free
    $needed = $total + $largest + $MarginBytes
    if ($free -lt $needed) { Exit-Fetch 4 "insufficient disk available=$free needed=$needed closure=$total staging=$largest margin=$MarginBytes" }
    Write-Log "PREFLIGHT RESULT=PASS available=$free needed=$needed closure=$total staging=$largest margin=$MarginBytes"
}

function Test-EffectiveHost([Uri]$Uri) {
    $host = $Uri.Host.ToLowerInvariant()
    return $Uri.Scheme -eq 'https' -and ($host -eq 'huggingface.co' -or $host -eq 'cdn-lfs.huggingface.co' -or $host -like '*.xethub.hf.co' -or $host -like '*.cdn.hf.co')
}

function Download-Partial([string]$Url, [string]$Partial, [string]$Journal, $Entry) {
    $start = Get-Length $Partial
    $lastError = $null
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        try {
            $requestedUri = [Uri]$Url
            if (-not $TestBaseUrl -and -not (Test-EffectiveHost $requestedUri)) { throw "redirect host refused effective_url=$requestedUri" }
            $handler = [Net.Http.HttpClientHandler]::new()
            $handler.AllowAutoRedirect = $true
            $handler.MaxAutomaticRedirections = 8
            $client = [Net.Http.HttpClient]::new($handler)
            $request = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, $Url)
            if ($start -gt 0) { $request.Headers.Range = [Net.Http.Headers.RangeHeaderValue]::Parse("bytes=$start-") }
            $response = $client.SendAsync($request, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
            if (-not $response.IsSuccessStatusCode) { throw "HTTP status $([int]$response.StatusCode)" }
            if ($start -gt 0 -and $response.StatusCode -ne [Net.HttpStatusCode]::PartialContent) { throw 'server refused secure Range resume' }
            $effectiveUri = $response.RequestMessage.RequestUri
            if (-not $TestBaseUrl -and -not (Test-EffectiveHost $effectiveUri)) {
                Write-Log "REDIRECT_HOST_REFUSED phase=post-transfer activation=refused effective_url=$effectiveUri"
                throw "redirect host refused effective_url=$effectiveUri"
            }
            $input = $response.Content.ReadAsStream()
            $output = [IO.File]::Open($Partial, [IO.FileMode]::Append, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try {
                $buffer = New-Object byte[] 1048576
                $nextLog = [DateTime]::UtcNow
                while (($read = $input.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $output.Write($buffer, 0, $read)
                    if ([DateTime]::UtcNow -ge $nextLog) { Write-Log "PROGRESS file=$($Entry.Name) bytes=$($output.Length) expected_bytes=$($Entry.Length)"; $nextLog = [DateTime]::UtcNow.AddSeconds(5) }
                }
                $output.Flush($true)
            } finally { $output.Dispose(); $input.Dispose(); $response.Dispose(); $client.Dispose(); $handler.Dispose() }
            return
        } catch {
            $lastError = $_.Exception.Message
            if ($attempt -lt 4) { Start-Sleep -Seconds $attempt }
        }
    }
    Exit-Fetch 3 "network failure url=$Url attempts=4 detail=$lastError journal=$Journal"
}

function Download-One($Entry, $Entries) {
    Assert-Preflight $Entries
    $url = if ($TestBaseUrl) { $TestBaseUrl.TrimEnd('/') + '/' + $Entry.Name } else { "$DefaultBase/$Revision/$($Entry.Name)" }
    $final = Join-Path $Dest $Entry.Name
    $partial = "$final.partial"
    $journal = Join-Path $script:JournalDir ($Entry.Name + '.journal')
    Write-Log "START file=$($Entry.Name) expected_bytes=$($Entry.Length) url=$url"
    if (Test-Path -LiteralPath $final) {
        if (Test-VerifiedFile $final $Entry.Length $Entry.Sha256) { Write-Log "DECISION=HIT file=$($Entry.Name)"; return }
        Quarantine $final 'existing-file-verification-failed'
    }
    Ensure-PartialAndJournal $partial $journal $url $Entry
    $start = Get-Length $partial
    if ($start -gt $Entry.Length) {
        Quarantine $partial 'partial-longer-than-expected'
        Quarantine $journal 'partial-longer-than-expected'
        Ensure-PartialAndJournal $partial $journal $url $Entry
        $start = 0
    }
    Write-Log ("DECISION={0} file={1} existing_bytes={2} expected_bytes={3}" -f $(if ($start -gt 0) {'RESUME'} else {'MISS'}), $Entry.Name, $start, $Entry.Length)
    Download-Partial $url $partial $journal $Entry
    if (-not (Test-VerifiedFile $partial $Entry.Length $Entry.Sha256)) { Exit-Fetch 1 "digest/length verification failure file=$($Entry.Name) expected_bytes=$($Entry.Length) expected_sha256=$($Entry.Sha256)" }
    [IO.File]::Move($partial, $final)
    Write-Log "COMPLETE file=$($Entry.Name) observed_bytes=$($Entry.Length) sha256=$($Entry.Sha256)"
}

if ($Revision -notmatch '^[a-f0-9]+$') { Write-Usage; exit 2 }
if (-not $Dest) { $Dest = Join-Path $env:LOCALAPPDATA ("franken_nlp\source\$Model\$Revision") }
if ($Revision -ne $DefaultRevision) {
    if (-not $AllowUntrustedRevision -or -not $Catalog) { Write-Log "UNTRUSTED_REVISION_REFUSED revision=$Revision requires -AllowUntrustedRevision and -Catalog"; exit 2 }
    Write-Log "UNTRUSTED_REVISION revision=$Revision catalog=$Catalog; this download is not the default recipe/catalog identity"
}
if ($TestBaseUrl -and $env:FNLP_FETCH_ALLOW_TEST_BASE_URL -ne '1') { Exit-Fetch 2 '-TestBaseUrl requires FNLP_FETCH_ALLOW_TEST_BASE_URL=1' }

Ensure-Destination
$entries = Get-Catalog
$script:LockDir = Join-Path $Dest ('.fnlp-fetch-lock-' + $Revision)
try { New-Item -ItemType Directory -LiteralPath $script:LockDir -ErrorAction Stop | Out-Null }
catch { Exit-Fetch 4 "revision lock busy or stale path=$script:LockDir; do not bypass concurrent access" }
Write-Log "LOCK RESULT=PASS path=$script:LockDir"

$failed = $false
$missingCount = 0
foreach ($entry in $entries) {
    if ($CheckOnly) {
        Write-Log "CHECK_ONLY file=$($entry.Name) expected_bytes=$($entry.Length)"
        $checkPath = Join-Path $Dest $entry.Name
        if (-not (Test-Path -LiteralPath $checkPath)) { $missingCount++ }
        if (-not (Test-VerifiedFile $checkPath $entry.Length $entry.Sha256)) { $failed = $true }
    } else { Download-One $entry $entries }
}
if ($failed) {
    if ($CheckOnly -and $missingCount -eq $entries.Count) { Write-Log "CHECK_ONLY RESULT=SKIPPED_NO_MODEL files=0/$($entries.Count)"; Exit-Fetch 0 }
    Write-Log "CHECK_ONLY RESULT=FAIL files=$($entries.Count)/$($entries.Count)"; Exit-Fetch 1
}
if ($CheckOnly) { Write-Log "CHECK_ONLY RESULT=PASS files=$($entries.Count)/$($entries.Count)" }
else { Write-Log "FETCH_MODEL RESULT=PASS files=$($entries.Count)/$($entries.Count) bytes=$(($entries | Measure-Object -Property Length -Sum).Sum)" }
[Console]::Error.WriteLine(('NEXT source="{0}" command="fnlp convert --source {0} --source-manifest docs/truth-pack/nanbeige4.2-3b.source.json --recipe nanbeige42-int8-v1 --arch generic -o nanbeige4.2-3b.fnlpq-v1.int8.generic.fnlpq"' -f $Dest))
Exit-Fetch 0
