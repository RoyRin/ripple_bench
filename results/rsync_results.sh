#!/bin/bash
# Example rsync script - update paths and credentials as needed

FROM=/path/to/remote/results
TO=.

rsync -azP user@cluster.example.edu:$FROM $TO
