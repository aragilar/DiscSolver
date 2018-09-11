#!/bin/sh

# from http://codeinthehole.com/writing/tips-for-using-a-git-pre-commit-hook/
git stash push --keep-index --include-untracked
./dev_hooks/run_tests.sh
RESULT=$?
git stash pop
[ $RESULT -ne 0 ] && exit 1
exit 0

