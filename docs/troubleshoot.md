# Troubleshooting

## when extracting binaries, got the following error about no permission to create a symlink

Error:
```
tar: slorado-e7c7e54/lib/libamdhip64.so.5: Cannot create symlink to ‘libamdhip64.so’: Operation not supported
tar: Exiting with failure status due to previous errors
```

Solution:
- try to extract to a different location that supports symbolic links
- or else, make a hard link. You can do this by `cd slorado-binary-dir/lib/ && ln libamdhip64.so libamdhip64.so.5`


# Getting an error that /tmp/something is unwritable

Error example:
```
MIOpen(HIP): Error [FlushUnsafe] File is unwritable: /tmp/gfx90a68.HIP.2_20_0_f185a6464-dirty.ufdb.txt
```

Solution:
```
mkdir /tmp/a_unique_name
export TMPDIR=/tmp/a_unique_name
./slorado .....
```

** IMPORTANT: Make sure you give an existent directory, we had a typo as  `a_unnique_name` when exporting, and torch was giving a segfault!
