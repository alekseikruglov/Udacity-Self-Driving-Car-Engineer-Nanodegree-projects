FILEID="1FpICY-diUIp6HsD4twVO6KX3HKF5qNiM"
FILENAME="trainingData"

cd /opt/

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILEID -O $FILENAME+".zip" && rm -rf /tmp/cookies.txt


unzip $FILENAME+".zip" -d $FILENAME

rm $FILENAME+".zip"