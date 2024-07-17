# mac系统

## finder使用：

1, 如何快速获取文件的绝对路径：command + option+c  
2, 

在finder中显示路径：
defaults write com.apple.finder _FXShowPosixPathInTitle -bool TRUE;killall Finder
如果要取消路径显示：defaults delete com.apple.finder _FXShowPosixPathInTitle;killall Finder



用户权限：
在mac中，“admin”用户组是一个特殊的用户组，包括具有管理员权限的用户账户
查看“admin”用户组
dscl . -read /Groups/admin GroupMembership  







## terminal的使用：

1，打开文件夹窗口：open .  //打开当前文件所在的文件夹
             open ～   open /home/



