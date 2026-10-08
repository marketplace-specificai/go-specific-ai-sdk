# Changelog

All notable changes to the SpecificAI Go SDK are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the module
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every release carries a **capability table** so a reader can answer "what can a
customer on this version actually do?" without downloading the module. The
Python SDK spent two months with `inference_url` in the repository and absent
from PyPI, and nobody noticed because no such table existed.

The rows below for published versions were verified by downloading the module
zips from `proxy.golang.org` and grepping them, not by reading this repository.
Module path: `github.com/marketplace-specificai/go-specific-ai-sdk`.

## [v0.1.0] - Unreleased

The Go SDK was **not** affected by the publish drift that motivated this
release cycle: `WithInferenceURL` has been in every published version since
v0.0.3, so Go customers have always had direct-Triton inference. Go is the
reference for what the other two SDKs should look like, not a victim.

The minor bump (planned as v0.0.5) is for the one behaviour change below: an
empty `BaseModelName` no longer means BERT.

### Changed

- `TrainingManager.Start` no longer defaults `StartParams.BaseModelName` to
  `"bert-base-uncased"`. Empty now means **Automatic**: the SDK omits
  `base_distilled_model` and `base_model_name`, and the trainer picks the base
  model (hyperparameter search for encoder tasks, the default decoder for
  summarization and generation). A non-empty value pins that model, and an
  unsupported one fails the training with an explanatory status message.
  Callers that relied on the old default and want BERT must now set
  `BaseModelName: "bert-base-uncased"` explicitly. v0.0.4 and earlier still
  send it, and the platform now treats it as a pinned choice rather than a
  placeholder.
- `Start` sends `task_type` on the training payload too.

### Added

- `TestWithInferenceURL` in `client_test.go`: asserts the option is honoured
  verbatim as the Triton base, that a client built with `WithInferenceURL` alone
  (no `WithBaseURL`) can run inference, and that omitting it derives
  `{gatewayRoot}/public/triton` instead.
- This changelog.

### Capabilities

| Capability | Supported |
| --- | --- |
| Platform automation — tasks, assets, trainings, models | Yes |
| Gateway inference — `WithBaseURL` | Yes |
| Direct Triton inference — `WithInferenceURL` | Yes (since v0.0.3) |
| Tracing of SpecificAI model calls — `WithTrace` | Yes |
| Logging of external (non-SpecificAI) model calls — `LogClassification` / `LogSummarization` / `LogEntityRecognition` | Yes |
| Task types — classification, NER, generation | Yes |
| OpenAI and Anthropic wrappers | Yes |
| Session-cookie auth — `WithSessionCookie` | Yes |
| Automatic base-model selection — empty `BaseModelName` | **Yes — first release with it** |
| Published to the Go module proxy | **No — pending** |

Publishing v0.1.0 is **not** automatic. `.github/workflows/go_sdk_publish.yml`
syncs to `specific-ai-org/specificai-go-sdk`, which is neither the module path in
`go.mod` nor a repository that currently resolves; its only run to date
(2026-06-22) exited early because `GO_SDK_MIRROR_DEPLOY_KEY` is unset. Cut this
release with `./publish.sh v0.1.0` until that automation is repointed.

## [v0.0.4] - 2026-06-22

Documentation and import-path corrections only; no behaviour change from v0.0.3.

### Fixed

- `INSTALL.md` and `integration_test.go` referenced the old module paths
  (`github.com/specific-ai-org/specificai-go-sdk` and
  `github.com/optune-ai/optune/go-sdk`). Both now use
  `github.com/marketplace-specificai/go-specific-ai-sdk`, so copy-pasted
  `go get` and import lines resolve.

### Capabilities

| Capability | Supported |
| --- | --- |
| Platform automation — tasks, assets, trainings, models | Yes |
| Gateway inference — `WithBaseURL` | Yes |
| Direct Triton inference — `WithInferenceURL` | Yes |
| Tracing of SpecificAI model calls — `WithTrace` | Yes |
| Logging of external (non-SpecificAI) model calls | Yes |
| Task types — classification, NER, generation | Yes |
| OpenAI and Anthropic wrappers | Yes |
| Session-cookie auth — `WithSessionCookie` | Yes |
| Published to the Go module proxy | Yes — current `@latest` |

## [v0.0.3] - 2026-06-22

The release that gave Go customers direct-Triton inference. This is the
capability the Python and JS SDKs still owe their users.

### Added

- `WithInferenceURL` — direct Triton inference, bypassing the gateway. When
  unset, the SDK derives `{gatewayRoot}/public/triton` from `WithBaseURL`.
- `LogClassification`, `LogSummarization`, and `LogEntityRecognition` for
  recording external (non-SpecificAI) model calls into the platform, plus the
  `examples/log` example.
- `INSTALL.md`, and integration tests for the client, inference, and HTTP layers.
- CI workflow (`.github/workflows/build.yml`) with the auto patch-bump publish
  job.

### Capabilities

| Capability | Supported |
| --- | --- |
| Platform automation — tasks, assets, trainings, models | Yes |
| Gateway inference — `WithBaseURL` | Yes |
| Direct Triton inference — `WithInferenceURL` | **Yes — first release with it** |
| Tracing of SpecificAI model calls — `WithTrace` | Yes |
| Logging of external (non-SpecificAI) model calls | **Yes — first release with it** |
| Task types — classification, NER, generation | Yes |
| OpenAI and Anthropic wrappers | Yes |
| Session-cookie auth — `WithSessionCookie` | Yes |
| Published to the Go module proxy | Yes |

## [v0.0.2] - 2026-06-18

### Added

- `publish.sh`, the manual release script still used to cut versions today.

### Capabilities

Identical to v0.0.1 — no library code changed.

## [v0.0.1] - 2026-06-18

First published release, cut from the branch behind PR #2126 before it merged.

### Added

- Unified `Client` with functional options (`WithBaseURL`, `WithAPIKey`,
  `WithTimeout`, `WithSessionCookie`, `WithTrace`).
- Platform managers: tasks, assets, trainings, models.
- Gateway inference through `{gatewayRoot}/public/triton`, with classification,
  NER, and generation response parsing.
- Async tracing collector, and OpenAI and Anthropic wrappers.

### Capabilities

| Capability | Supported |
| --- | --- |
| Platform automation — tasks, assets, trainings, models | Yes |
| Gateway inference — `WithBaseURL` | Yes |
| Direct Triton inference — `WithInferenceURL` | **No — added in v0.0.3** |
| Tracing of SpecificAI model calls — `WithTrace` | Yes |
| Logging of external (non-SpecificAI) model calls | **No — added in v0.0.3** |
| Task types — classification, NER, generation | Yes |
| OpenAI and Anthropic wrappers | Yes |
| Session-cookie auth — `WithSessionCookie` | Yes |
| Published to the Go module proxy | Yes |
