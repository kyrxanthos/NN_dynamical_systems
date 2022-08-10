#!/bin/bash
for FILE in $1/Plots/*.pdf; do
  pdfcrop "${FILE}"
  rm "${FILE}"
done