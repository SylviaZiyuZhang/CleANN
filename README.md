Forked from https://github.com/microsoft/DiskANN, commit #35f8cf7


# Commands
Fast incremental builds can be done with
```
pip3 install --no-build-isolation -ve .
```
If you only change a .cc file, this should be relatively fast.

To develop in vscode and have syntax highlighting work (tested with the clangd extension), run
```
scripts/generate-compile-commands.sh
```
