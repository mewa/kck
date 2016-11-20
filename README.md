# kck

bibliotek do odczytywania kodów QR - qrtools

instalacja ->

  `brew install zbar`
  
  `pip install qrtools`
  
  (aby uniknac segfaulta przy `import qrtools` instalujemy zbar'a z patchem)
  
  `pip install git+https://github.com/npinchot/zbar.git`

# usage
`./sheet.py IMG_FNAME PAGE_MOMENT_FNAME QR_MOMENT_FNAME`

Copy&paste using provided moments:
`./sheet.py IMG page_moment.txt blank.txt`
