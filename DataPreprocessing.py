from typing import List
import config

# 删除非法字符
def remove_illegal_char(word:str)->str:
    # 定义栈，存放非法字符
    stack=list()
    # 存放结果
    result_chars=list()
    # 便利每一个字符
    for char in word:
        # 判断是不是开符号
        if char in config.OPEN_CHAR:
            stack.append(char) # 如果是开符号，加入栈
        elif char in config.CLOSE_CHAR_MAP.keys(): # 判断是不是闭符号
            # 判断栈最后一个符合是否和闭符号匹配
            if stack and stack[-1]==config.CLOSE_CHAR_MAP.get(char):
                stack.pop() # 匹配，删除stack里面最后一个符号
            else:
                pass
        # 将合法的符号加入result_chars
        else:
            if not stack:
                result_chars.append(char)
    # 返回结果
    return "".join(result_chars)

if __name__=="__main__":
    print(remove_illegal_char("a(b)c)"))