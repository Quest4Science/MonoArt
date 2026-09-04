# Publishing checkpoints

The 557 MiB model bundle is intentionally excluded from Git. Publish it as a GitHub Release asset together with its JSON manifest. [GitHub accepts individual release assets smaller than 2 GiB](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas), so `monoart_stage1.pt` fits without Git LFS.

## Maintainer upload

First ensure the code on `main` is current and authenticate an account with write access:

```bash
gh auth status
gh api repos/Quest4Science/MonoArt --jq '.permissions'
git push origin main
```

The permission report must show `push: true`. If it does not, switch to an authorized GitHub account or ask a repository administrator for write access before continuing.

Create and push an annotated version tag, then create the release from that exact tag:

```bash
git tag -a v0.1.0 -m "MonoArt v0.1.0"
git push origin v0.1.0

gh release create v0.1.0 \
  --repo Quest4Science/MonoArt \
  --verify-tag \
  --title "MonoArt v0.1.0" \
  --generate-notes \
  "checkpoints/monoart_stage1.pt#MonoArt Stage-1 checkpoint" \
  "checkpoints/monoart_stage1.pt.json#SHA-256 manifest"
```

If the release already exists but the assets have not been uploaded, use:

```bash
gh release upload v0.1.0 \
  --repo Quest4Science/MonoArt \
  checkpoints/monoart_stage1.pt \
  checkpoints/monoart_stage1.pt.json
```

Do not use `--clobber` unless replacing a published asset is intentional. Prefer a new version tag whenever checkpoint bytes change.

The equivalent browser workflow is **Releases → Draft a new release → choose the tag → attach both files → Publish release**. The command-line forms above are easier to audit; see the official [`gh release create`](https://cli.github.com/manual/gh_release_create) and [`gh release upload`](https://cli.github.com/manual/gh_release_upload) references.

## User download

Users can download a fixed version with the GitHub CLI:

```bash
mkdir -p checkpoints
gh release download v0.1.0 \
  --repo Quest4Science/MonoArt \
  --pattern 'monoart_stage1.pt*' \
  --dir checkpoints
monoart verify-checkpoint checkpoints/monoart_stage1.pt
```

See the official [`gh release download`](https://cli.github.com/manual/gh_release_download) reference for additional filters and archive options.

No GitHub account or CLI is required for a public release:

```bash
curl -L -o checkpoints/monoart_stage1.pt \
  https://github.com/Quest4Science/MonoArt/releases/download/v0.1.0/monoart_stage1.pt
curl -L -o checkpoints/monoart_stage1.pt.json \
  https://github.com/Quest4Science/MonoArt/releases/download/v0.1.0/monoart_stage1.pt.json
monoart verify-checkpoint checkpoints/monoart_stage1.pt
```

The `latest/download` URL is convenient for demos, while a versioned URL is preferable for reproducible experiments.
