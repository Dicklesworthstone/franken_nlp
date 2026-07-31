# OQ-12 GGUF-prior lineage

+## Official baseline

+The conversion prior is tied to official `ggml-org/llama.cpp` commit
+`000547513f1530346ecd163db8b3e13962949961`, selected by the baseline record.
+It is 57 commits after the minimum Nanbeige-support commit
+`b77d646751d01c0962bc203b6809e9d94f7d50b7`.  The mapping and tier rules
+are source observations at that immutable commit: [Nanbeige conversion]
+(https://raw.githubusercontent.com/ggml-org/llama.cpp/000547513f1530346ecd163db8b3e13962949961/conversion/nanbeige.py),
+[Llama mapping]
+(https://raw.githubusercontent.com/ggml-org/llama.cpp/000547513f1530346ecd163db8b3e13962949961/conversion/llama.py), and
+[quant tiers]
+(https://raw.githubusercontent.com/ggml-org/llama.cpp/000547513f1530346ecd163db8b3e13962949961/src/llama-quant.cpp).

+## Authors' fork

+`Nanbeige/llama.cpp` branch `nanbeige42` was observed at
+`c6640a1c0cf7b38df342b67021a3900b04d092e7` on 2026-07-31 via
+`git ls-remote https://github.com/Nanbeige/llama.cpp.git refs/heads/nanbeige42`.
+The branch tip is a reported observation, not a pin in the baseline record.
+No merge-base or source-diff evidence is retained here, so its divergence from
+the official baseline is **inconclusive**.  It is historical lineage only and
+never an independent oracle or a deciding vote.

+## Authority boundary

+No local GGUF exists for this audit, and no GGUF digest or converter tensor
+inventory is claimed.  The JSON tables are source-rule predictions/search
+seeds only.  Full candidate artifacts and held-out metrics decide any int4
+recipe; nearest GGUF formats are peers, never a "match" claim.
+