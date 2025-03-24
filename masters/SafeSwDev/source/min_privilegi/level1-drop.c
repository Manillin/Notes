#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <stdio.h>

int main(int argc, char **argv, char **envp)
{
  uid_t uid;
  uid = getuid();

  setresuid(-1, uid, -1);
  system("/usr/bin/env echo and now what?");
}
