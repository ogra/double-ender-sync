# Licensing guidance for source-only distribution

This project is MIT licensed, but it depends on third-party components with their own licenses.

## Distribution model

Current policy:

- Distribute source code only.
- Do not publish prebuilt binaries from GitHub Actions or GitHub Releases.
- End users install dependencies in their own environment.

This model reduces legal complexity compared to binary redistribution.

## Third-party licenses to review

At minimum, review and keep notices for:

- PySide6 / Qt for Python (LGPL/commercial options from Qt Company)
- soundfile (BSD-3-Clause)
- libsndfile (LGPL-2.1-or-later)
- numpy, scipy and other transitive dependencies

## Recommended compliance steps

1. Keep `LICENSE` (MIT) in repository root.
2. Maintain `THIRD_PARTY_NOTICES.md` with dependency names, license IDs, and upstream links.
3. Keep CI workflows focused on tests/lint only (no packaged binary artifacts).
4. If distribution policy changes and binaries are published later, perform legal review first and add binary-redistribution obligations documentation.

## Trigger points for legal re-review

Perform a new review if any of the following occur:

- Publishing executable bundles/installers
- Shipping Docker images to external users
- Including native runtime libraries in releases
- Changing GUI framework or audio backend dependencies
