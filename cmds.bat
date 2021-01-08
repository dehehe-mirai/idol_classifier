REM py3.7
virtualenv target
cmd
REM activate
target/target/Scripts/activate
pip install tflite_runtime-2.5.0-cp37-cp37m-win_amd64.whl
deactivate

pip install pyinstaller
pyinstaller -F --add-data "./Lib/site-packages/mxnet/*.dll;./mxnet" .\predict.py
upx-3.96-win64\upx.exe -9 dist\predict.exe