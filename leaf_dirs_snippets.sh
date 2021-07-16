# this only works on Linux
find ~/sarnobat.git/mwk/snippets/  -type d -links 2 | perl -pe 's{/home/sarnobat}{\$HOME}g' | nc localhost 2000