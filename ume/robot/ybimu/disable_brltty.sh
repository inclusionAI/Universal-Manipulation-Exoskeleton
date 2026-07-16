# Disable BRLTTY

sudo systemctl stop brltty
sudo systemctl mask brltty

sudo mkdir -p /etc/udev/rules.d/backup
sudo mv /lib/udev/rules.d/*brltty* /etc/udev/rules.d/backup/