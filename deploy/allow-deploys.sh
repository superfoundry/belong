#!/bin/sh
# Let GitHub deploy main to this VM. Run once, from the project folder, after deploy/setup.sh, with
# the PUBLIC half of the deploy key (the line in belong_deploy.pub):
#   sudo sh deploy/allow-deploys.sh "ssh-ed25519 AAAA... belong-github-deploy"
# That key logs in as the user "deploy" and can only run belong-update (replace the app's code and
# restart it): no shell, no forwarding, no other sudo.
set -eu
KEY="${1:?Give the public key: the line from belong_deploy.pub}"
case "$KEY" in
  ssh-ed25519\ *|ecdsa-sha2-*|ssh-rsa\ *) ;;
  *) echo "That doesn't look like a public key (it should start with ssh-ed25519)." >&2; exit 1 ;;
esac
install -m 755 -o root -g root deploy/belong-update /usr/local/sbin/belong-update
id deploy >/dev/null 2>&1 || useradd --create-home --shell /bin/sh deploy
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
printf 'restrict,command="sudo /usr/local/sbin/belong-update" %s\n' "$KEY" > /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh/authorized_keys
chmod 600 /home/deploy/.ssh/authorized_keys
printf 'deploy ALL=(root) NOPASSWD: /usr/local/sbin/belong-update\n' > /etc/sudoers.d/belong-deploy
chmod 440 /etc/sudoers.d/belong-deploy
visudo -cf /etc/sudoers.d/belong-deploy
echo "Done. GitHub can now deploy main here (and do nothing else)."
