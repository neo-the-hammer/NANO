// ======================================================================
//  Copyright (c) 2012, G. Fiori, P. Marconcini, G. Iannaccone  University of Pisa
//  
//  This file is released under the BSD license.
//  See the file "license.txt" for information on usage and
//  redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
// ====================================================================== 

#include "writeout.h"
static PyObject* py_writeout(PyObject* self, PyObject* args)
{

  char *string2;
  PyObject *string;
  import_array();

  if (!PyArg_ParseTuple(args,"O",&string))
    {
      printf("NOT AN ARRAY! \n");
      exit(0);
    }
  /* The callers also pass arrays and numbers (writeout(self.skparameters),
     writeout(bound)); anything that is not a str is printed through str(). */
  #if PY_MAJOR_VERSION >=3
  PyObject *text = PyObject_Str(string);
  if (text == NULL)
    return NULL;
  string2=(char *)PyUnicode_AsUTF8(text);
  if (string2 == NULL)
    {
      Py_DECREF(text);
      return NULL;
    }
  #else
  string2=PyString_AsString(string);
  #endif
  printf("%s", string2);
  printf("\n");
  #if PY_MAJOR_VERSION >=3
  Py_DECREF(text);
  #endif
  char *s = "Hello from writeout!";
  return Py_BuildValue("s", s);
}
  
