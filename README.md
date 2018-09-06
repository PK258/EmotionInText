# Installing weights  
cd ~/Senti/Senti_v2/model/       
curl -L -o deepmoji_weights.hdf5 https://www.dropbox.com/s/bb9y0hpk939k9bp/deepmoji_weights.hdf5?dl=0  

cd ..  
python3 score_texts_emojis.py 'i love you'  
