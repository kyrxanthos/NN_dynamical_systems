#!/bin/bash
for FILE in ./Plots/*.pdf; do
  pdfcrop "${FILE}"
  rm "${FILE}"
done