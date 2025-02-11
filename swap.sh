#!/bin/bash
echo "Activando swap..."
fallocate -l 512M /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "Swap activado correctamente."