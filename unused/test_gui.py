from tkinter import *
import time
import random 
def gui():
 
  ###******回调函数定义******###
  def sendMsg():                  #发送消息
    strMsg = 'lumos:' + time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime()) + '\n '
    txtMsgList.insert(END, strMsg, 'greencolor')   #插入到tag位置
    txtMsgList.insert(END, txtMsg.get('0.0', END))
    line="test"
    txtText.delete('0.0',END)
    txtText.insert(END,line)
  def cancelMsg():                #取消消息
    txtMsg.delete('0.0', END)
  def clearMsg():
    txtMsgList.delete('0.0',END)
    txtMsg.delete('0.0',END)
  def sendMsgEvent(event):        #发送消息事件
    if event.keysym == "Return":  #按回车键可发送
      sendMsg()
  ###******回调函数定义******###
  #创建窗口 
  t = Tk()
  t.title('小e聊天机器人')     # 窗口名称
  t.geometry('1080x750')
  t.resizable(0, 0)           # 禁止调整窗口大小
  ###******创建frame容器******###
  frmB0 = Frame(t,width=500,height=30)
  frmB1 = Frame(t,width=500, height=500)
  frmB2 = Frame(t,width=500, height=50)
  frmB3 = Frame(t,width=500, height=50)
  #第三列
  frmC11= Frame(t,width=700, height=700)
  ###******创建frame容器******###

  ###******创建控件******###
  #1.Text控件
  txtMsgList = Text(frmB1)                          #frmB1表示父窗口
  #创建并配置标签tag属性
  txtMsgList.tag_config('greencolor',               #标签tag名称
                        foreground='#008C00')       #标签tag前景色，背景色为默认白色
 
  txtMsg = Text(frmB2);
  txtMsg.bind("<KeyPress-Return>", sendMsgEvent)    #事件绑定，定义快捷键
 
#   timeText=Text(frmC2,font=("Times", "28", "bold italic"),height=1,bg="PowderBlue")
#   timeText2=Text(frmC2,fg="blue",font=("Times", "12","bold italic"))
 
  txtText=Text(frmC11,font=("Times", "11",'bold'),  #字体控制
               width=200,height=100,                  #文本框的宽（in characters ）和高(in lines) (not pixels!)
               spacing2=7,                          #文本的行间距
               bd=2,                                #边框宽度
               padx=5,pady=5,                       #距离文本框四边的距离
               selectbackground='blue',             #选中文本的颜色
               state=NORMAL)                        #文本框是否启用 NORMAL/DISABLED
                                                    # insert(插入位置，插入内容)
  #2.Button控件
  btnSend = Button(frmB3, text='发送', width = 8,cursor='heart', command=sendMsg)
  btnCancel = Button(frmB3, text='取消', width = 8,cursor='shuttle', command=cancelMsg)
  btnClear=Button(frmB0,text='清空',width=8,cursor='heart',command=clearMsg)
  ###******创建控件******###
 
 
  ###******窗口布局******###
  frmB0.grid(row=0, column=0, columnspan=1, rowspan=1, padx=1, pady=1)
  frmB1.grid(row=1, column=0, columnspan=1, rowspan=1, padx=1, pady=1)
  frmB2.grid(row=2, column=0, columnspan=1, padx=1, pady=1)
  frmB3.grid(row=3, column=0, columnspan=1, padx=1)
  
  frmC11.grid(row=0, column=1, rowspan=3, padx=1, pady=1)  
 
  ###******窗口布局******###
  #固定大小
  frmB0.grid_propagate(0)
  frmB1.grid_propagate(0)
  frmB2.grid_propagate(0)
  frmB3.grid_propagate(0)
  frmC11.grid_propagate(0)
 
 
  ###******控件布局******### 
 
  btnSend.grid(row=0, column=0)
  btnCancel.grid(row=0, column=1)
  btnClear.grid(row=0,column=0)
  txtMsgList.grid(row=1,column=0)
  txtMsg.grid()

  txtText.grid(row=1,column=0,pady=5)


  #主事件循环
  t.mainloop()
 
if __name__ == '__main__':
    gui()