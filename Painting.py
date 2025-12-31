import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
from matplotlib import cm
import os
import time

def validator(function):
    def wrapper(epoch:int,title:str,save:bool,**kwargs):
        # 创建保存的目录
        if save:
            os.makedirs("Drawing_Results",exist_ok=True)
        names = list(kwargs.keys())  # loss tag
        losses = list(kwargs.values())
        # 如果传入的losses为空
        if len(losses) == 0:
            logger.error(f"losses is empty!")
            exit(1)
        losses_len = list(map(len, losses))
        # 判断losses中所有的loss长度是否相等
        if len(set(losses_len)) > 1:
            logger.error(f"The lengths of the loss lists are not equal!")
            exit(1)
        # 判断loss长度是否和epoch相等
        if losses_len[0] != epoch:
            logger.error(f"The length of the loss list does not match the number of epochs!")
            exit(1)
        function(epoch,title,save,**kwargs)
    return wrapper

@ validator
def plot_loss_curve(epoch:int,title:str,save:bool,**losses):
    plt.figure()
    plt.title(label=title)
    names=list(losses.keys()) # loss tag
    losses=list(losses.values())
    epoch_list=np.linspace(1,epoch,epoch,dtype=np.int32)
    # 循环所有的数据
    for name,loss in zip(names,losses):
        plt.plot(epoch_list,loss,label=f"{name}")
    plt.xticks(ticks=np.arange(start=1,stop=epoch,step=2))
    plt.legend()
    if save:
        if not os.path.exists(f"./Drawing_Results/{title}.png"):
            plt.savefig(f"./Drawing_Results/{title}.png")
        else:
            file_name=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
            plt.savefig(f"./Drawing_Results/{file_name}.png")
    plt.show()

@ validator
def loss_curve_comparison(epoch:int,title:str,save:bool,**losses):
    plt.figure()
    names=list(losses.keys()) # loss tag
    losses=list(losses.values())
    epoch_list=np.linspace(1,epoch,epoch,dtype=np.int32)
    colors=cm.tab10(range(epoch))
    np.random.shuffle(colors)
    for index,(name,loss) in enumerate(zip(names,losses)):
        # 目前只支持一行二列的摆放，最多支持两组数据
        axes=plt.subplot(1,2,index+1)
        axes.plot(epoch_list,loss,color=colors[index],label=f"{title.lower()}")
        axes.set_title(name)
        axes.set_xticks(np.arange(1,epoch,2))
        axes.legend()
    if save:
        # 建议文件名使用时间来表示，防止重复
        if not os.path.exists(f"./Drawing_Results/{title}.png"):
            plt.savefig(f"./Drawing_Results/{title}.png")
        else:
            file_name=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
            plt.savefig(f"./Drawing_Results/{file_name}.png")
    plt.show()

if __name__=="__main__":
    epoch = 10
    train_loss = [0.89, 0.75, 0.65, 0.58, 0.52, 0.47, 0.43, 0.40, 0.37, 0.35]
    val_loss = [0.92, 0.78, 0.68, 0.61, 0.55, 0.50, 0.46, 0.42, 0.39,0.22]
    plot_loss_curve(epoch,"WER",save=False,wer=train_loss)
    plot_loss_curve(epoch=epoch, title="LOSS", save=False, train_loss=train_loss, val_loss=val_loss)