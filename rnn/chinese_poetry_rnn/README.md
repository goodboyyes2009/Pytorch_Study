# Use rnn to generate Chinese Poetry
---

### 中文繁体转简体
```bash
# ubuntu
 sudo apt-get install cconv
```

### 繁转简脚本
```bash
#!/bin/bash
for ff in `ls *.json`
do
    cconv -f utf8-tw -t UTF8-CN $ff -o simplified/$ff
done
```