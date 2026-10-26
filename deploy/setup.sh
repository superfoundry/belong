#!/bin/sh
# Set up (or update) the hosted demo on a Debian or Ubuntu VM. Run from the unpacked project folder:
#   sudo sh deploy/setup.sh demo.example.com          (a domain pointing at the VM)
#   sudo sh deploy/setup.sh 34-1-2-3.sslip.io         (no domain: the VM's IP with dashes)
# Your OpenAI key and the demo password go in /opt/belong/.env, which this never overwrites.
set -eu
ADDRESS="${1:?Give the address people will open, e.g. 34-1-2-3.sslip.io}"

apt-get update -q
apt-get install -y -q python3 caddy

id belong >/dev/null 2>&1 || useradd --system --home-dir /opt/belong --shell /usr/sbin/nologin belong
mkdir -p /opt/belong
rm -rf /opt/belong/belong /opt/belong/out
cp -r belong README.md SPEC.md TRUST.md /opt/belong/
if [ ! -f /opt/belong/.env ]; then
  printf 'OPENAI_API_KEY=\nBELONG_PASSWORD=\n' > /opt/belong/.env
  NEW_ENV=1
fi
chown -R belong:belong /opt/belong
chmod 600 /opt/belong/.env

cp deploy/belong.service /etc/systemd/system/belong.service
sed "s/BELONG_ADDRESS/$ADDRESS/" deploy/Caddyfile > /etc/caddy/Caddyfile
systemctl daemon-reload
systemctl enable belong caddy
systemctl restart belong
systemctl reload caddy || systemctl restart caddy

if [ "${NEW_ENV:-}" = 1 ]; then
  echo "Next: sudo nano /opt/belong/.env   (add your OpenAI key and a demo password), then: sudo systemctl restart belong"
fi
echo "The demo will be at https://$ADDRESS once ports 80 and 443 are open to the VM."
