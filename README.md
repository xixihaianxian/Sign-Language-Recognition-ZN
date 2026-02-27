## 服务器上的修改

- ### 1. 代码
  ```python
  # DataPreprocessing.py

  def handle_words(words: List[str]) -> List[str]:
    new_words =[]
    for i in range(len(words)):
        word = words[i]
        word = remove_illegal_char(word)
        
        # 🚨 核心修改：暴力过滤所有中英文标点
        invalid_chars =['。', '，', '？', '！', '、', '；', '：', '.', ',', '?', '!', '“', '”', '（', '）']
        if word in invalid_chars:
            continue  # 遇到标点，直接丢弃！
            
        if word[-1].isdigit() and not word[0].isdigit():
            word = word[:-1]
            
        if word.isdigit():
            word = str(int(word))
            
        if len(word) > 0:
            new_words.append(word)
            
    return new_words
  ```