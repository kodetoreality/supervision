# Release Process

This doc outlines how supervision is released into production.

It assumes you already have the code changes, as well as a draft of the release notes.

1. Make sure you have all required changes were merged into `develop`.
2. Create and merge a PR, merging `develop` into `master`, containing:
   - A commit that updates the project version in `pyproject.toml`.
   - All changes made during the release.
3. Tag the commit with the new supervision version.
   - Upon pushing the tag, the [PyPi](https://pypi.org/project/supervision/) should update to the new version. Check this!
4. Open and merge a PR, merging `master` into `develop`.
5. Update the docs by running the [Supervision Release Documentation Workflow 📚](https://github.com/roboflow/supervision/actions/workflows/publish-release-docs.yml) workflow from GitHub.
    - Select the `master` branch from the dropdown.
6. Create a release on GitHub.
   - Go to releases
   - Assign the release notes to the tag created in step 3.
   - Publish the release.