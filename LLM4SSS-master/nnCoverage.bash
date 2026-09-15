dataset="human"
model="GPT4SSS"
python ./nnCoverage/getData.py -C ./conf/$dataset/$model.json
python ./nnCoverage/nnCoverage.py -C ./conf/$dataset/$model.json
