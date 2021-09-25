sudo apt install -y zip unzip wget
export fileid=1tp7efLhznpLwXuUjc_X92Lv3HCSBRyE3
export filename=ARS408_ros.zip

wget --save-cookies /tmp/cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > /tmp/confirm.txt

wget --load-cookies /tmp/cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(</tmp/confirm.txt)
unzip ARS408_ros.zip
rm -rf /tmp/cookies.txt /tmp/confirm.txt ./ARS408_ros.zip