After installing atari_py, we still require to install the ROM.

follow instructions [here](https://github.com/openai/atari-py#roms)

make sure you have ```unrar``` install.

```bash
sudo apt install unrar
```

```bash
$ wget http://www.atarimania.com/roms/Roms.rar
$ unrar e Roms.rar
$ python -m atari_py.import_roms .
```
