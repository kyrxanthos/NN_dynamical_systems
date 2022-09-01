#!/bin/bash
BASEDIR=$(dirname $0)
echo "Script location: ${BASEDIR}"

for FILE in $BASEDIR/*.pdf; do
  pdfcrop "${FILE}"
  rm "${FILE}"
done