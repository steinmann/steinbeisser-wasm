# GitHub Pages Deployment

This repository is set up to deploy the site in `dist/` to GitHub Pages with GitHub Actions.

## What is already configured

- GitHub Actions workflow: `.github/workflows/deploy.yml`
- Static build script: `scripts/build.sh`
- Pages artifact output: `dist/`
- `.nojekyll` is created automatically by the build script
- The workflow installs the exact `wasm-bindgen-cli` version that matches `engine/Cargo.lock`

## Before you start

GitHub Pages from a private repository is only available on plans that allow private-repository Pages.

- Personal account: GitHub Pro or higher
- Organization: GitHub Team, GitHub Enterprise Cloud, or higher

The site itself is public on the internet unless you are using the GitHub Enterprise Cloud private Pages visibility feature.

## First deployment checklist

1. Create a GitHub repository for this folder.
2. Keep the repository private.
3. Push this project to the repository's default branch.
4. In the repository settings, set Pages to use GitHub Actions as the source.
5. Let the `Deploy Pages` workflow finish.
6. Open the Pages URL shown in the repository Pages settings.

## Local verification

From the repository root:

```bash
./scripts/build.sh
python3 ./scripts/serve.py dist
```

Then open:

- `http://127.0.0.1:4173`

## Notes

- The workflow only deploys from the repository's default branch.
- Manual workflow runs also only deploy when run against the default branch.
- `dist/` and `engine/target/` are intentionally ignored and do not need to be committed.
