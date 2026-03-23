# Release Checklist

Follow these steps every time you want to publish a new version to PyPI.

## 1. Sync shared files from FIDDLE (if needed)

If `dataset.py`, `model_tcn.py`, or any `utils/` files were updated in FIDDLE, the sync workflow will have opened a PR in this repo automatically. **Review and merge that PR before proceeding.**

## 2. Update the version

In `setup.py`, bump the version number:

```python
version="0.2.0",  # e.g. 0.1.0 → 0.2.0
```

Follow [semantic versioning](https://semver.org/):
- `0.0.x` — bug fixes
- `0.x.0` — new features, backwards compatible
- `x.0.0` — breaking changes

## 3. Update CHANGELOG.md

Add a new entry at the top of `CHANGELOG.md`:

```markdown
## [0.2.0] - YYYY-MM-DD
### Added
- ...
### Changed
- ...
### Fixed
- ...
```

## 4. Commit and push to main

```bash
git add setup.py CHANGELOG.md
git commit -m "Release v0.2.0"
git push origin main
```

## 5. Create a GitHub Release

Go to the `msfiddle` repo on GitHub → **Releases** → **Draft a new release**:

1. Click **"Choose a tag"** → type `v0.2.0` → **"Create new tag: v0.2.0 on publish"**
2. Set title to `v0.2.0`
3. Paste the CHANGELOG entry into the description
4. Click **Publish release**

This will automatically trigger the PyPI publish workflow.

## 6. Verify

- Check the **Actions** tab in the repo to confirm the workflow succeeded
- Check [pypi.org/project/msfiddle](https://pypi.org/project/msfiddle/) to confirm the new version is live
- Test the new release locally:

```bash
pip install msfiddle==0.2.0
```

## 7. Update FIDDLE (if model weights changed)

If new checkpoint files (`.pt`) were added:
1. Create a release in the [FIDDLE repo](https://github.com/JosieHong/FIDDLE) with a tag matching the `msfiddle` version (e.g. `v2.0.0`) and attach the new `.pt` files

`msfiddle` automatically derives the FIDDLE release tag from the installed package version — no manual update to `download.py` is needed.