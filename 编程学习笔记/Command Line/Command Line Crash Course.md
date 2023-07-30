# Introduction: Shut Up and Shell

-  Shut up and type all of this in

# hostname - What's Ur PC's Name?

    hostname

-  This is the name of your computer, or at least one of its names.

# mkdir - Make A Directory

    mkdir temp
    mkdir temp/stuff
    mkdir temp/stuff/things
    mkdir -p temp/stuff/things/frank/joe/alex/john

-  **mkdir **, it makes directories (folders)
-  Using the slash(/) character for all paths
-  Windows also use the backslash(\) characters

# pwd - Paths, Folders, Directories

    pwd

-  **pwd** means "printing working directory"

# cd - Change Directory

    cd temp
    pwd
    cd stuff
    pwd
    cd ../../..
    cd .
    cd \

-  use the .. to move up
-  \  return to the root directory

# ls - List Directory

    cd temp
    ls

-  **ls** command lists out the contents of the directory you are currently in.
-  On Windows, **dir** can do the same

# rmdir - Remove Directory

    cd temp/stuff/things/frank/joe/alex/john
    cd ..
    rmdir john
    cd .. 
    rmdir alex
    cd .. 
    ls

# pushd, popd - Moving Around

-  These commands let you temporarily go to a different directory and then come back, easily switching between the two
-  **pushd** command takes your current directory and pushes it into a list for later, the it changes to another directory
-  **popd** command takes the last directory you pushed and pops it off, taking you back there

# touch, New-Item - Making Empty Files

    New-Item iamcool.txt -type file

-  On Unix, touch
-  On Windows, New-Item

# cp - Copy A File

    cd temp
    cp iamcool.txt neat.txt
    cp neat.txt awesome.txt
    mkdir something
    cp awesome.txt something/
    ls something/
    cp -r something newplace
    ls newplace/

# mv - Moving A File

    cd temp
    mv awesome.txt uncool.txt
    ls

-  Moving files or, rather, renaming them
	-  just give the old name and the new name

# less, more - View A File

    more neat.txt

-  This is one way to look at the contents of a file. If the file has many lines, it will page so that only one screenful at a time is visible

# cat - Stream A File

    cat neat.txt

-  This command just spews the whole file to the screen with no paging or stopping

# rm - Removing A File

    cd temp
    ls
    rm uncool.txt
    ls
    rm iamcool.txt neat.txt 

# Pipes and Redirection

    cd temp
    ls
    cat ex12.txt > ex13.txt

-  Now we get to the cool part of the command line: *redirection*
-  The concept is that you can take a command and you can change where its input and output goes
-  Use < (less-than)  >(greater-than)  | (pipe)  symbols to do this
-  |  -  takes the output from the command on the left , and "pipes" it to the command on the right
-  <  -  will take and send the input from the file on the right to the program on the left. But does not work in PowerShell
-  >  -  takes the output of the command on the left, then writes it to the file on the right
-  >>  -  takes the output of the command on the left, the appends it to the file on the right

# Wildcard Matching

    cd temp
    ls *.txt
    ls ex*.*
    ls e*
    ls *t

-   \*  asterisk symbol, to say 'anything'
-   Wherever you put the asterisk, the shell will build a list of all the files that match the non-asterisk part

# find, DIR -R  -  Finding Files

    dir -r
    dir -r -filter "*.txt"

-  find (dir -r on Windows)  command to search for all files ending in .txt

# grep, select-string  -  Looking Inside Files

     echo > newfile.txt
     位于命令管道位置 1 的 cmdlet Write-Output
     请为以下参数提供值:
     InputObject[0]: This is a new file
     InputObject[1]: This is a new file
     InputObject[2]: This is a new file
     InputObject[3]:
     
	  echo > oldfile.txt
	  位于命令管道位置 1 的 cmdlet Write-Output
	  请为以下参数提供值:
	  InputObject[0]: This is an old file
	  InputObject[1]: This is an old file
	  InputObject[2]: This is an old file
	  InputObject[3]:
	  
	  select-string new *.txt
	  newfile.txt:1:This is a new file
	  newfile.txt:2:This is a new file
	  newfile.txt:3:This is a new file
	  
	  select-string old *.txt
	  oldfile.txt:1:This is an old file
	  oldfile.txt:2:This is an old file
	  oldfile.txt:3:This is an old file

-  select-string on Windows

# man, HELP  -  Getting Command Help

    help dir

- Use the help command in Windows to find information about commands

# HELP -  Finding Help

    help *Dir*

-  Sometimes you forget the name of a command but you know what it does. This command looks through all the help files and finds potentially relevant help for you

# env, echo, Env:  -  What's In Your Environment

    echo $PWD
    $env:TEMP
    $env:OS

# export, Env:  -  Changing Environment Variables

    $env:TESTING="anaconda boo"
    $env:TESTTING
    remove-item Env:\TESTING

# exit  -  Exitiing Your Terminal
    