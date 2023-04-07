## Instructions for running multicore on EC2 via Jupyter

#### Initial installation

* Step 1: Rent EC2 on AWS. Maybe c5.24xlarge? With Amazon Linux 2 AMI.

* Step 2: From local: connect to EC2 via `ssh -i [PEMFILE].pem ec2-user@[EC2-IP].compute-1.amazonaws.com`

* Step 3: On EC2, run

```
mkdir future-assessment-model
sudo yum groupinstall "Development Tools"
sudo yum install htop tmux zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel libpcap-devel xz-devel yum-utils nss-devel libffi-devel wget
```

* Step 4: Install Python 3.9, since Squigglepy doesn't work on earlier versions of Python (3.7) included by default:

```
sudo yum-builddep python3
cd /opt
sudo wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz 
sudo tar xzf Python-3.9.6.tgz

# Fix SSL for Python install
mkdir /tmp/openssl
cd /tmp/openssl
wget https://www.openssl.org/source/openssl-1.0.2q.tar.gz
tar xvf openssl-1.0.2q.tar.gz
cd /tmp/openssl/openssl-1.0.2q
./config
make
sudo make install
cd /opt/Python-3.9.6

# At this point you must `vim Modules/Setup`, search for SSL, and uncomment lines

sudo ./configure --enable-optimizations
sudo make
sudo make install
cd ..
sudo rm -f /opt/Python-3.9.6.tgz

cd ~/future-assessment-model
python3.9 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
```

* Step 5: From local, within this folder: Use `scp` to upload files: `scp -r -i [PEMFILE].pem *.ipynb *.txt *.py caches modules ec2-user@[EC2-IP].compute-1.amazonaws.com:~/future-assessment-model/.`

* Step 6: From EC2, run Jupyter:

```
screen
jupyter lab --no-browser
```

* Step 7: From local, open up a forwarding port to Jupyter: `ssh -i [PEMFILE.pem -L 8000:localhost:8888 ec2-user@[EC2-IP].compute-1.amazonaws.com`

* Step 8: Go to URL indicated by Jupyter from Step 6, but on port 8000 (via Step 7) instead of 8888 (the default).
