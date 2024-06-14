#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2015, IBM Corp.
# All rights reserved.
#
# Distributed under the terms of the BSD Simplified License.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

"""
idaDataFrame
---------
An efficient 2D container that looks like a panda's DataFrame and behave the
same, the only difference is that it uses only a reference to a remote database
instead of having the data loaded into memory

Also similar to its R counterpart, data.frame, except providing automatic data
alignment and a host of useful data manipulation methods having to do with the
labeling information
"""

# Ensure Python 2 compatibility
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import dict
from builtins import zip
from builtins import str
from builtins import int
from future import standard_library
standard_library.install_aliases()

import sys
import os
from copy import deepcopy
import warnings
from numbers import Number
from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.core.index import Index, RangeIndex

from lazy import lazy
import six

import ibmdbpy
import ibmdbpy.statistics
import ibmdbpy.indexing
import ibmdbpy.aggregation
import ibmdbpy.filtering
import ibmdbpy.utils

from ibmdbpy.utils import timed, chunklist
from ibmdbpy.internals import InternalState
from ibmdbpy.exceptions import IdaDataFrameError
from ibmdbpy.internals import idadf_state


class IdaDataFrame(object):
    """
    An IdaDataFrame object is a reference to a table in a remote Db2 Warehouse
    database. IDA stands for In-DataBase Analytics. IdaDataFrame copies the
    Pandas interface for DataFrame objects to ensure intuitive interaction for
    end-users.

    Examples
    --------
    >>> idadb = IdaDataBase('BLUDB') # See documentation for IdaDataBase
    >>> ida_iris = IdaDataFrame(idadb, 'IRIS')
    >>> ida_iris.cov()
                      sepal_length  sepal_width  petal_length  petal_width
    sepal_length      0.685694    -0.042434      1.274315     0.516271
    sepal_width      -0.042434     0.189979     -0.329656    -0.121639
    petal_length      1.274315    -0.329656      3.116278     1.295609
    petal_width       0.516271    -0.121639      1.295609     0.581006
    """

    # TODO: Check if everything is ok when selecting AND projecting with loc
    # TODO: BUG: Filtering after selection/projection

    def __init__(self, idadb, tablename, indexer = None):
        """
        Constructor for IdaDataFrame objects.

        Parameters
        ----------
        idadb : IdaDataBase
            IdaDataBase instance which contains the connection to be used.
        tablename : str
            Name of the table to be opened in the database.
            It should contain only alphanumeric characters and underscores.
            All lower case characters will be converted to upper case characters.
        indexer : str, optional
            Name of the column that should be used as an index. This is
            optional. However, if no indexer is given, the order of rows issued
            by the head and tail functions is not guaranteed. Also, several
            in-database machine learning algorithms need an indexer as a
            parameter to be executed.

        Attributes
        ----------
        _idadb : IdaDataBase
            IdaDataBase object parent of the current instance.
        tablename : str
            Name of the table self references.
        name : str
            Full name of the table self references, including schema.
        schema : str
            Name of the schema the table belongs to.
        indexer : str
            Name of the column used as an index. "None" if no indexer.
        loc : str
            Indexer that enables the selection and projection of IdaDataFrame
            instances. For more information, see the loc class documentation.
        internal_state : InternalState
            Object used to internally store the state of the IdaDataFrame. It
            also allows several non-destructive manipulation methods.
        type : str
            Type of the IdaDataFrame : “Table”, “View”, or “Unknown”.
        dtypes : DataFrame
            Data type in the database for each column.
        index : pandas.core.index
            Index containing the row names in this table.
        columns : pandas.core.index
            Index containing the columns names in this table.
        axes : list
            List containing columns and index attributes.
        shape : Tuple
            Number of rows and number of columns.

        Notes
        -----
        Attributes "type", "dtypes", "index", "columns", "axes", and "shape"
        are evaluated in a lazy way to avoid an overhead when creating an
        IdaDataFrame. Sometimes the index may be too big to be downloaded.

        Examples
        --------
        >>> idadb = IdaDataBase('BLUDB')
        >>> ida_iris = IdaDataFrame(idadb, "IRIS")
        """
        #TODO: Implement equality comparision between two IdaDataFrames

        if not idadb.__class__.__name__ == "IdaDataBase":
            idadb_class = idadb.__class__.__name__
            raise TypeError("Argument 'idadb' is of type %s, expected : IdaDataBase"%idadb_class)
        tablename = ibmdbpy.utils.check_tablename(tablename)

        #idadb._reset_attributes("cache_show_tables")

        # TODO: Test what kind of error append when a table in use and cached
        # is suddently deleted

        if idadb.exists_table_or_view(tablename) is False:
            # Try again after refreshing the cache
            idadb._reset_attributes("cache_show_tables")
            # Refresh the show table cache in parent IdaDataBase, because a table
            # could have been created by other mean / user and we have to make sure
            # the lookup is done and is actual.
            if idadb.exists_table_or_view(tablename) is False:
                raise NameError("Table %s does not exist in the database %s."
                                %(tablename, idadb.data_source_name))


        self._idadb = idadb
        self._indexer = indexer

        # Initialise indexer object
        self.loc = ibmdbpy.indexing.Loc(self)

        if "." in tablename:
            self.schema = tablename.split('.')[0]
            #self.name = tablename.split('.')[-1]
            self._name = tablename
            self.tablename = tablename.split('.')[-1]
        else:
            self.schema = idadb.current_schema
            self._name = idadb.current_schema + '.' + tablename
            self.tablename = tablename

        # self._name is the original name, this is a "final" variable

        # Push a reference to itself in its parent IdaDataBase
        self._idadb._idadfs.append(self)
        # TODO : self.size

        # A cache for unique value of each column
        self._unique = dict()

###############################################################################
### Attributes & Metadata computation
###############################################################################

    @lazy
    def internal_state(self):
        """
         InternalState instances manage the state of an IdaDataFrame instance
         and allow several non-destructive data manipulation methods, such as
         the selection, projection, filtering, and aggregation of columns.

        """
        return InternalState(self)


    @property
    @idadf_state
    def name(self):
        return self.internal_state.current_state


    @property
    def indexer(self):
        """
        The indexer attribute refers to the name of a column that should be
        used to index the table. This makes sense because Db2 Warehouse is a
        column-based database, so row IDs do not make sense and are not
        deterministic. As a consequence, the only way to address a particular
        row is to refer to it by its index. If no indexer is provided, ibmdbpy
        still works but a correct row order is not guaranteed as far as the
        dataset is not sorted. Also, note that the indexer column is not taken
        into account in data mining algorithms.
        """
        if hasattr(self, "_indexer"):
            return self._indexer
        else:
            None    

        
    @indexer.setter
    def indexer(self, value):
        """
        Basic checks for indexer :
        * The column exists in the table.
        * All values are unique.
        """
        if value is None:
            return

        if value not in self.columns:
            raise IdaDataFrameError("'%s' cannot be used as indexer "%value +
                                    " because this is not a column in '%s'"%self._name)

        del self.columns
        #count = self[value].count_distinct() ## TODO: TO FIX, should return directly just a number
        count = self.levels(value)
        if count < self.shape[0]:
            raise IdaDataFrameError("'%s' cannot be used as indexer "%value +
                                    " because it contains non unique values.")
        self._indexer = value


    @lazy
    def type(self):
        """
        Type of self: 'Table', 'View'  or 'Unknown'.

        Returns
        -------
        str
            idaDataFrame type.

        Examples
        --------
        >>> ida_iris.type
        'Table'
        """
        return self._get_type()

    @lazy
    @idadf_state(force = True)
    def dtypes(self):
        """
        Data type in database for each column in self.

        Returns
        -------
        DataFrame
            In-Database type for each columns.

        Examples
        --------
        >>> ida_iris.dtypes
                     TYPENAME
        sepal_length   DOUBLE
        sepal_width    DOUBLE
        petal_length   DOUBLE
        petal_width    DOUBLE
        species       VARCHAR
        """
        #import pdb ; pdb.set_trace()
        return self._get_columns_dtypes()

    @lazy
    @idadf_state
    # to deprecate
    def index(self):
        """
        Index containing the row names in self.

        Returns
        -------
        Index

        Examples
        --------
        >>> ida_iris.index
        Int64Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
            ...
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149],
           dtype='int64', length=150)

        Notes
        -----
        Because indexes in a database can be only numeric, it is not that
        interesting for an IdaDataFrame but can still be useful sometimes. The
        function can break if the table is too large. Ask for the user’s
        approval before downloading an index which has more than 10000 values.
        """
        return self._get_index()

    @lazy
    def columns(self):
        """
        Index containing the column names in self.

        Returns
        -------
        Index

        Examples
        --------
        >>> ida_iris.columns
        Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
        'species'],
        dtype='object')
        """
        if hasattr(self, "internal_state"):
            self.internal_state._create_view()
            cols = self._get_columns()
            self.internal_state._delete_view()
            return cols
        else:
            return self._get_columns()

    @lazy
    @idadf_state
    # to deprecate (no index)
    def axes(self):
        """
        List containing IdaDataFrame.columns and IdaDataFrame.index attributes.

        Returns
        -------
        list
            List containing two indexes (indexes and column attributes).

        Examples
        --------
        >>> ida_iris.axes
        [Int64Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
             ...
             140, 141, 142, 143, 144, 145, 146, 147, 148, 149],
            dtype='int64', length=150),
        Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
        'species'],
        dtype='object')]
        """
        return [self.index, self.columns]

    @lazy
    @idadf_state
    def shape(self):
        """
        Tuple containing number of rows and number of columns.

        Returns
        -------
        tuple

        Examples
        --------
        >>> ida_iris.shape
        (150, 5)
        """
        return self._get_shape()

    @property
    @idadf_state
    def empty(self):
        """
        Boolean that is True if the table is empty (no rows).

        Returns
        -------
        Boolean
        """
        if self.shape[0] == 0:
            return True
        else:
            return False

    def __len__(self):
        """
        Number of records.

        Returns
        -------
        int

        Examples
        --------
        >>> len(idadf)
        150
        """
        return self.shape[0]

    def __iter__(self):
        """
        Iterate over columns.
        """
        return iter(self.columns)

    def __getitem__(self, item):
        """
        Enable label based projection (selection of columns) in IdaDataFrames.

        Enable slice based selection of rows in IdaDataFrames.

        Enable row filtering.

        The syntax is similar to Pandas.

        Examples
        --------

        >>> idadf['col1'] # return an IdaSeries

        >>> idadf[['col1']] # return an IdaDataFrame with one column

        >>> idadf[['col1', 'col2', 'col3']] # return an IdaDataFrame with 3 columns

        >>> idadf[0:9] # Select the 10 first rows

        >>> idadf[idadf['col1'] = "test"]
        # select of rows for which attribute col1 is equal to "test"

        Notes
        -----
        The row order is not guaranteed if no indexer is given and the dataset
        is not sorted
        """

        if isinstance(item, ibmdbpy.filtering.FilterQuery):
            newidadf = self._clone()
            newidadf.internal_state.update(item)
            newidadf._reset_attributes(["shape"])
        else:
            if isinstance(item, slice):
                return self.loc[item]
            if not (isinstance(item,six.string_types)|isinstance(item, list)):
                raise KeyError(item)
            if isinstance(item, six.string_types):
                # Case when only one column was selected
                if item not in self.columns:
                    raise KeyError(item)
                newidaseries = self._clone_as_serie(item)

                # Form the new columndict
                for column in list(newidaseries.internal_state.columndict):
                    if column != item:
                        del newidaseries.internal_state.columndict[column]
                newColumndict = newidaseries.internal_state.columndict

                # Erase attributes
                newidaseries._reset_attributes(["columns", "shape", "dtypes"])
                # Set columns and columndict attributes
                newidaseries.internal_state.columns = ["\"%s\""%col for col in item]
                newidaseries.internal_state.columndict = newColumndict
                # Update, i.e. appends an entry to internal_state._cumulative
                newidaseries.internal_state.update()

                # Performance improvement
                # avoid, caused wrong dtypes for the result
                # newidaseries.dtypes = self.dtypes.loc[[item]]

                return newidaseries

            # Case of multiple columns
            not_a_column = [x for x in item if x not in self.columns]
            if not_a_column:
                raise KeyError("%s"%not_a_column)

            newidadf = self._clone()

            # Form the new columndict
            newColumndict = OrderedDict()
            for col in item:
                # Column name as key, its definition as value
                newColumndict[col] = self.internal_state.columndict[col]

            # Erase attributes
            newidadf._reset_attributes(["columns", "shape", "dtypes"])
            # Set columns and columndict attributes
            newidadf.internal_state.columns = ["\"%s\""%col for col in item]
            newidadf.internal_state.columndict = newColumndict
            # Update, i.e. appends an entry to internal_state._cumulative
            newidadf.internal_state.update()

            # Performance improvement
            # avoid, caused wrong dtypes for the result
            # newidadf.dtypes = self.dtypes.loc[item]

        return newidadf

    def __setitem__(self, key, item):
        """
        Enable the creation and aggregation of columns.

        Examples
        --------

        >>> idadf['new'] = idadf['sepal_length'] * idadf['sepalwidth']
        # select a new column as the product of two existing columns

        >>> idadf['sepal_length'] = idadf['sepal_length'] / idadf['sepal_length'].mean()
        # modify an existing column
        """
        if not (isinstance(item, IdaDataFrame)):
            raise TypeError("Modifying columns is supported only using "+
                            "IdaDataFrames.")
        if isinstance(key, six.string_types):
            key = [key]
        if len(key) != len(item.columns):
                raise ValueError("Wrong number of items passed %s, placement implies %s"%(len(item.columns),len(key)))

        #form the new columndict
        for newname, oldname in zip(key, item.columns):
            self.internal_state.columndict[newname] = item.internal_state.columndict[oldname]
        newColumndict = self.internal_state.columndict

        #erase attributes
        self._reset_attributes(["columns", "shape", "dtypes"])
        #set columns and columndict attributes
        self.internal_state.columndict = newColumndict
        self.internal_state.columns = ["\"%s\""%col for col in newColumndict.keys()]
        #update, i.e. appends an entry to internal_state._cumulative
        self.internal_state.update()

        # Flush the "unique" cache
        for column in key:
            if column in self._unique:
                del self._unique[column]

    def __delitem__(self, item):
        """
        Enable non-destructive deletion of columns using a Pandas style syntax.
        This happens inplace, which means that the current IdaDataFrame is
        modified.

        Examples
        --------
        >>> idadf = IdaDataFrame(idadb, "IRIS")
        >>> idadf.columns
        Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'], dtype='object')
        >>> del idadf['sepal_length']
        >>> idadf.columns
        Index(['sepal_width', 'petal_length', 'petal_width', 'species'], dtype='object')
        """
        if not (isinstance(item,six.string_types)):
            raise TypeError
        if item not in self.columns:
            raise KeyError(item)

        # Flush the "unique" cache
        if item in self._unique:
            del self._unique[item]

        self._idadb.delete_column(self, item, destructive = False)
        return

    def __enter__(self):
        """
        Allow the object to be used with a "with" statement
        """
        return self

    def __exit__(self):
        """
        Allow the object to be used with a "with" statement. Make sure that
        allow possible views related to the IdaDataFrame with be deleted when
        the object goes out of scope
        """
        while self.internal_state.viewstack:
            try :
                view = self.internal_state.viewstack.pop()
                # Just a last check to make sure not to drop user's db
                if view != self.tablename:
                    drop = "DROP VIEW \"%s\"" %view
                    self._prepare_and_execute(drop, autocommit = True)
            except: pass

    #We decided not to allow columns access idadf.columnname like this for now.
    #We could decide to allow it but for this we may have to switch all
    #existing attributes to private ("_" as first character) so to avoid
    #conflicts between attributes and columns because for example IdaDataFrame
    #has an attribute "name" -> if a column is labelled "name" this will make
    #it unavailable. Pandas do also poorly manage this issue, for example :
    #DataFrame attribute "size".

    #May be implemented in the future

    #def __getattr__(self, name):
    #    """
    #    After regular attribute access, try look up the name
    #    This allows simpler access to columns for interactive use.
    #    """
        # Note: obj.x will always call obj.__getattribute__('x') prior to
        # calling obj.__getattr__('x').
        #if hasattr(self, name):
        #    return object.__getattribute__(self,name)
        #else:
    #    if name not in self.__dict__["columns"]:
            #return self[name]
    #        return object.__getattribute__(self, name)
    #    raise AttributeError("'%s' object has no attribute '%s'" %
    #                         (type(self).__name__, name))

    #def __setattr__(self, name, value):
    #    if name not in self.__dict__["columns"]:
    #        self.__dict__[name] = value
    #    else:
    #        raise ValueError("It is not allowed to set the value of a column in an IdaDataFrame.")

    def __lt__(self, value):
        """
        ibmdbpy.filtering.FilterQuery object when comparing self using "<".
        """
        return ibmdbpy.filtering.FilterQuery(self.columns, self._name, "lt", value)

    def __le__(self, value):
        """
        ibmdbpy.filtering.FilterQuery object when comparing self using "<=".
        """
        return ibmdbpy.filtering.FilterQuery(self.columns, self._name, "le", value)

    def __eq__(self, value):
        """
        ibmdbpy.filtering.FilterQuery object when comparing self using "==".
        """
        return ibmdbpy.filtering.FilterQuery(self.columns, self._name, "eq", value)

    def __ne__(self, value):
        """
        ibmdbpy.filtering.FilterQuery object when comparing self using "!=".
        """
        return ibmdbpy.filtering.FilterQuery(self.columns, self._name, "ne", value)

    def __ge__(self, value):
        """
        ibmdbpy.filtering.FilterQuery object when comparing self using ">=".
        """
        return ibmdbpy.filtering.FilterQuery(self.columns, self._name, "ge", value)

    def __gt__(self, value):
        """
        ibmdbpy.filtering.FilterQuery object when comparing self using ">".
        """
        return ibmdbpy.filtering.FilterQuery(self.columns, self._name, "gt", value)

    ################ Arithmetic operations

    def __add__(self, other):
        """
        Perform an addition between self and another IdaDataFrame or number.

        Examples
        --------
        >>> ida = idadf['sepal_length'] + 3

        Notes
        -----
        Arithmetic operations only make sense if self contains only numeric columns.
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(self, "add", other)

    def __radd__(self, other):
        """
        Enable the reflexivity of the addtion operation.

        Examples
        --------
        >>> ida = idadf['sepal_length'] + 3

        >>> ida = 3 + idadf['sepal_length']
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(other, "add", self, swap = True)

    def __div__(self, other):
        """
        Perform a division between self and another IdaDataFrame or number.

        When __future__.division is not in effect.

        Examples
        --------
        >>> ida = idadf['sepal_length'] / 3

        Notes
        -----
        Arithmetic operations only make sense if self contains only numeric columns.
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(self, "div", other)

    def __rdiv__(self, other):
        """
        Enable the reflexivity of the division operation.

        When __future__.division is not in effect.

        Examples
        --------
        >>> ida = idadf['sepal_length'] / 3

        >>> ida = 3 / idadf['sepal_length']
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(other, "div", self, swap = True)

    def __truediv__(self, other):
        """
        Perform a division between self and another IdaDataFrame or number.

        When __future__.division is in effect.

        Examples
        --------
        >>> ida = idadf['sepal_length'] / 3

        Notes
        -----
        Arithmetic operations only make sense if self contains only numeric columns.
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(self, "div", other)

    def __rtruediv__(self, other):
        """
        Enable the reflexivity of the division operation.

        When __future__.division is in effect.

        Examples
        --------
        >>> ida = idadf['sepal_length'] / 3

        >>> ida = 3 / idadf['sepal_length']
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(other, "div", self, swap = True)

    def __floordiv__(self,other):
        """
        Perform an integer division between self and another IdaDataFrame or number.

        Examples
        --------
        >>> ida = idadf['sepal_length'] // 3

        Notes
        -----
        Arithmetic operations only make sense if self contains only numeric columns.
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(self, "floordiv", other)

    def __rfloordiv__(self, other):
        """
        Enable the reflexivity of the integer division operation.

        Examples
        --------
        >>> ida = idadf['sepal_length'] // 3

        >>> ida = 3 // idadf['sepal_length']
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(other, "floordiv", self, swap = True)

    def __mod__(self,other):
        """
        Perform a modulo operation between self and another IdaDataFrame or number.

        Examples
        --------
        >>> ida = idadf['sepal_length'] % 3

        Notes
        -----
        Arithmetic operations make sense if self has only numeric columns.
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(self, "mod", other)

    def __rmod__(self, other):
        """
        Enable the reflexivity of the modulo operation.

        Examples
        --------
        >>> ida = idadf['sepal_length'] % 3

        >>> ida = 3 % idadf['sepal_length']
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(other, "mod", self, swap = True)

    def __mul__(self,other):
        """
        Perform a multiplication between self and another IdaDataFrame or number.

        Examples
        --------
        >>> ida = idadf['sepal_length'] * 3

        Notes
        -----
        Arithmetic operations only make sense if self contains only numeric columns.
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(self, "mul", other)

    def __rmul__(self, other):
        """
        Enable the reflexivity of the multiplication operation.

        Examples
        --------
        >>> ida = idadf['sepal_length'] % 3

        >>> ida = 3 % idadf['sepal_length']
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(other, "mul", self, swap = True)

    def __neg__(self):
        """
        Calculate the absolute negative of all columns in self.

        Notes
        -----
        Arithmetic operations only make sense if self contains only numeric columns.
        """
        other = None
        return ibmdbpy.aggregation.aggregate_idadf(self, "neg", other)

    def __rpos__(self,other):
        """
        Calculate the absolute positive. No operation required.
        """
        return self

    def __pow__(self,other):
        """
        Perform a power operation between self and another IdaDataFrame or number.

        Examples
        --------
        >>> ida = idadf['sepal_length'] ** 3

        Notes
        -----
        Arithmetic operations only make sense if self contains only numeric columns.
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(self, "pow", other)

    def __rpow__(self, other):
        """
        Enable the reflexivity of the power operation.

        Examples
        --------
        >>> ida = idadf['sepal_length'] ** 3

        >>> ida = 3 ** idadf['sepal_length']

        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(other, "pow", self, swap = True)

    def __sub__(self,other):
        """
        Perform a substraction between self and another IdaDataFrame or number.

        Examples
        --------
        >>> ida = idadf['sepal_length'] - 3

        Notes
        -----
        Arithmetic operations only make sense if self contains only numeric columns.
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(self, "sub", other)

    def __rsub__(self, other):
        """
        Enable the reflexivity of the substraction operation.

        Examples
        --------
        >>> ida = idadf['sepal_length'] - 3

        >>> ida = 3 - idadf['sepal_length']
        """
        self._combine_check(other)
        return ibmdbpy.aggregation.aggregate_idadf(other, "sub", self, swap = True)

    #def __truediv__(self,value): ########
    #    pass

    #def __concat__(self,value): ####
    #    pass

    #def __contains__(self,value):
    #    pass

    # TODO: Do the inplace versions (with "i" in front)

###############################################################################
### Database Features
###############################################################################

    ##############################
    ## Delegation from IdaDataBase
    ##############################

    def exists(self):
        """
        Convenience function delegated from IdaDataBase.

        Check if the data still exists in the database.
        """
        return self._idadb.exists_table_or_view(self._name)

    def is_view(self):
        """
        Convenience function delegated from IdaDataBase.

        Check if the IdaDataFrame corresponds to a view in the database.
        """
        if self.type == 'View':
            return True
        else:
            return False

    def is_table(self):
        """
        Convenience function delegated from IdaDataBase.

        Check if the IdaDataFrame corresponds to a table in the database.
        """
        if self.type == 'Table':
            return True
        else:
            return False

    def get_primary_key(self):
        # TODO: What happen if the primary key is composed of several columns ?
        """
        Get the name of the primary key of self, if there is one. Otherwise,
        this function returns 0. This function may be deprecated in future
        versions because it is not very useful.
        """
        name = self.internal_state.current_state

        pk = self.ida_query("SELECT NAME FROM SYSIBM.SYSCOLUMNS WHERE TBNAME = '" +
                            name + "' AND KEYSEQ > 0 ORDER BY KEYSEQ ASC", first_row_only = True)
        if pk:
            return pk[0]
        else:
            return False

    # Should we maybe allow this only in IdaDataBase object ?
    #@timed
    def ida_query(self, query, silent = False, first_row_only = False, autocommit = False):
        """
        Convenience function delegated from IdaDataBase.

        Prepare, execute and format the result of a query in a data frame or in
        a tuple. See the IdaDataBase.ida_query documentation.
        """
        return self._idadb.ida_query(query, silent, first_row_only, autocommit)

    def ida_scalar_query(self, query, silent = False, autocommit = False):
        """
        Convenience function delegated from IdaDataBase.

        Prepare and execute a query and return only the first element as a
        string. See the IdaDataBase.ida_scalar_query documentation.

        """
        return self._idadb.ida_scalar_query(query, silent, autocommit)



###############################################################################
### Data Exploration
###############################################################################

    def print(self):
        print(self.internal_state.get_state())

    @idadf_state
    def head(self, nrow=5, sort=True):
        """
        Print the n first rows of the instance, n is set to 5 by default.

        Parameters
        ----------
        nrow : int > 0
            Number of rows to be included in the result.

        sort: default is True
            If set to True and no indexer is set the data will be
            sorted by the first numeric column or if no numeric column
            is available by the first column of the dataframe.
            If set to False and no indexer is set the row order is not
            guaranteed and can vary with each execution. For big tables
            this option might save query processing time.


        Returns
        -------
        DataFrame or Series
            The index of the corresponding row number and the columns are all
            columns of self. If the IdaDataFrame has only one column, it
            returns a Series.


        Examples
        --------
        >>> ida_iris.head()
           sepal_length  sepal_width  petal_length  petal_width species
        0           5.1          3.5           1.4          0.2  setosa
        1           4.9          3.0           1.4          0.2  setosa
        2           4.7          3.2           1.3          0.2  setosa
        3           4.6          3.1           1.5          0.2  setosa
        4           5.0          3.6           1.4          0.2  setosa
        """
        if (nrow < 1) | (not isinstance(nrow, int)):
            raise ValueError("Parameter nrow should be an int greater than 0.")
        else:
            name = self.internal_state.current_state

            order = ''
            if not " ORDER BY " in self.internal_state.get_state():
                if (self.indexer is not None)&(self.indexer in self.columns):
                    order = " ORDER BY \"" + str(self.indexer) + "\" ASC"
                elif self.indexer is None:
                    if sort:
                        column = self.columns[0]
                        if self._get_numerical_columns():
                            column = self._get_numerical_columns()[0]
                        order = " ORDER BY \"" + column + "\" ASC"
                    else:
                        order = ''
            data = self.ida_query("SELECT * FROM %s%s FETCH FIRST %s ROWS ONLY"%(name, order, nrow))

            if data.shape[0] != 0:
                # otherwise column sort order is reverted
                if not 'SELECT ' in name:
                    columns = self.columns
                    data.columns = columns
#                data = ibmdbpy.utils._convert_dtypes(self, data)
                if isinstance(self, ibmdbpy.IdaSeries):
                    data = pd.Series(data)

            return data

    # TODO : There is a warning in anaconda when there are missing values -> why ?
    @idadf_state
    def tail(self, nrow=5, sort=True):
        """
        Print the n last rows of the instance, n is set to 5 by default.

        Parameters
        ----------
        nrow : int > 0
            The number of rows to be included in the result.

        sort: default is True
            If set to True and no indexer is set the data will be
            sorted by the first numeric column or if no numeric column
            is available by the first column of the dataframe.
            If set to False and no indexer is set the row order is not
            guaranteed and can vary with each execution. For big tables
            this option might save query processing time.

        Returns
        -------
        DataFrame
            The index of the corresponding row number and the columns are all
            columns of self.


        Examples
        --------
        >>> ida_iris.tail()
             sepal_length  sepal_width  petal_length  petal_width    species
        145           6.7          3.0           5.2          2.3  virginica
        146           6.3          2.5           5.0          1.9  virginica
        147           6.5          3.0           5.2          2.0  virginica
        148           6.2          3.4           5.4          2.3  virginica
        149           5.9          3.0           5.1          1.8  virginica
        """
        if (nrow < 1) | (not isinstance(nrow, int)):
            raise ValueError("Parameter nrow should be an int greater than 0.")
        else:
            column_string = '\"' + '\", \"'.join(self.columns) + '\"'

            name = self.internal_state.current_state

            if " ORDER BY " in self.internal_state.get_state():
                query = "SELECT * FROM %s FETCH FIRST %s ROWS ONLY"%(name, nrow)
                data = self.ida_query(query)
                data.columns = self.columns
                data.set_index(data[self.indexer], inplace=True)
            else:
                order = ''
                if self.indexer:
                    sortkey = str(self.indexer)
                    order = "ORDER BY \"" + sortkey + "\""
                else:
                    if sort:
                        sortkey = self.columns[0]
                        if self._get_numerical_columns():
                            sortkey = self._get_numerical_columns()[0]
                        order = "ORDER BY \"" + sortkey + "\""
                query = ("SELECT * FROM (SELECT * FROM (SELECT " + column_string +
                         ", ((ROW_NUMBER() OVER(" + order +
                         "))-1) AS ROWNUMBER FROM " + name +
                         ") ORDER BY ROWNUMBER DESC FETCH FIRST " + str(nrow) +
                         " ROWS ONLY) ORDER BY ROWNUMBER ASC")
                data = self.ida_query(query)
                data.set_index(data.columns[-1], inplace=True)  # Set the index
                data.columns = self.columns

            data.index.name = None
            #            data = ibmdbpy.utils._convert_dtypes(self, data)
            if isinstance(self, ibmdbpy.IdaSeries):
                    data = pd.Series(data[data.columns[0]])
            return data

    @idadf_state
    def pivot_table(self, values=None, columns=None, max_entries=1000,
                    sort=None, factor_threshold=20, interactive=False,
                    aggfunc='count'):
        """
        Compute an aggregation function over all rows of each column that is
        specified as a value on the dataset. The result grouped by the columns
        defined in “columns”.

        Parameters
        ----------
        values: str or list or str optional
            List of columns on which “aggfunc” is computed.
        columns: str or list or str optional
            List of columns that is used as an index and by which the
            dataframe is grouped.
        max_entries: int, default=1000
            The maximum number of cells to be part of the output. By default,
            set to 1000.
        sort: str, optional
            Admissible values are: “alpha” and “factors”.
                * If “alpha”, the index of the output is sorted according to the alphabetical order.
                * If “factors”, the index of the output will be sorted according to increasing number of the distinct values.

            By default, the index will be sorted in the same order that is specified in “columns” argument.

        factor_threshold: int, default: 20
            Number of distinct values above which a categorical column should
            not be considered categorical anymore and under which a numerical
            column column should not be considered numerical anymore.
        interactive: bool
            If True, the user is asked if he wants to display the output, given
            its size.
        aggfunc: str
            Aggregation function to be computed on each column specified in the
            argument “values”. Admissible values are: “count”, “sum”, “avg”.
            This entry is not case-sensitive.

        Returns
        -------
        Pandas Series with Multi-index (columns)

        Examples
        --------
        >>> val = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        >>> ida_iris.pivot_table(values= val, aggfunc="avg")
                      species
        sepal_length  setosa        5.006
                      versicolor    5.936
                      virginica     6.588
        sepal_width   setosa        3.428
                      versicolor    2.770
                      virginica     2.974
        petal_length  setosa        1.462
                      versicolor    4.260
                      virginica     5.552
        petal_width   setosa        0.246
                      versicolor    1.326
                      virginica     2.026
        dtype: float64
        """
        # TODO : Support index
        from ibmdbpy.statistics import pivot_table
        return pivot_table(idadf=self, values=values, columns=columns,
                           max_entries=max_entries, sort=sort,
                           factor_threshold=factor_threshold,
                           interactive=interactive, aggfunc=aggfunc)

    @idadf_state
    def groupby(self, by):
        # TODO: create an IdadataFrame groupby
        raise NotImplementedError()

    @idadf_state
    def merge(self):
        raise NotImplementedError()

    @idadf_state
    def concat(self):
        raise NotImplementedError()

    @idadf_state
    def join(self):
        raise NotImplementedError()

    # TODO : implement NULL FIRST option
    @idadf_state
    def sort(self, columns=None, axis=0, ascending=True, inplace=False):
        """
        Sort the IdaDataFrame row wise or column wise.

        Parameters
        ----------
        columns : str or list of str
            Columns that should be used to sort the rows in the IdaDataFrame.
            If columns is set to None and axis to 0, then the IdaDataFrame
            columns are sorted in lexicographical order.
        axis : int (0/1)
             Axis that is sorted. 0 for sorting row wise, 1 for sorting column
             wise.
        ascending : bool, default: True
            Sorting order, True : ascending, False : descending
        inplace : bool, default: False
            The current object is modified or creates a modified copy. If
            False, the function creates a modified copy of the current
            dataframe. If True,  the function modifies the current dataframe.

        Raises
        ------
        ValueError
            * When sorting by column (column not None), the axis value must be 0 (rows).
            * A column does not belong to self.
            * The axis argument has a value other than 0 or 1.

        Notes
        -----
        If columns is set to None and axis to 0, this undoes all sorting the
        IdaDataFrame and returns the original sorting in the Db2 Warehouse
        database.

        No actual changes are made in Db2 Warehouse, only the querying changes.
        Everything is registered in an InternalState object. Changes can be
        observed by using  head and tail function.
        """
        if isinstance(columns, six.string_types):
            columns = [columns]

        # Sanitiy check
        if columns:
            if axis != 0:
                raise ValueError("When sorting by column, axis must be 0 (rows)")
            for column in columns:
                if column not in self.columns:
                    raise ValueError("Column "+column+" is not in "+self._name)
        if axis not in [0,1]:
            raise ValueError("Allowed values for axis is 0 (rows) or 1 (columns)")

        if inplace:
            idadf = self
        else:
            idadf = self._clone()

        if axis:
            # Sort the columns in ascending or descending lexicographic order
            idadf.internal_state.columndict = OrderedDict(sorted(idadf.internal_state.columndict.items(), reverse = not ascending))
            idadf.internal_state.update()
        else:
            if columns:
                idadf.internal_state.set_order(columns, ascending)
                idadf.internal_state.update()
            else:
                if isinstance(idadf, ibmdbpy.IdaSeries):
                    idadf.internal_state.set_order([idadf.column], ascending)
                    idadf.internal_state.update()
                else:
                    idadf.internal_state.reset_order()

        if not inplace:
            return idadf

    @idadf_state
    def levels(self, columns = None):
        # TODO: Test, doc, name?
        """
        Return the numbers of distinct values
        """
        if columns is not None:
            if isinstance(columns, six.string_types):
                columns = [columns]
            for column in columns:
                message = ''
                if column not in self.columns:
                    message += "Column %s does not belong to current idadataframe. \n"%column
                if message:
                    raise ValueError(message)
        else:
            columns = self.columns

        agglist = []
        for column in columns:
            agglist.append("COUNT(DISTINCT \"%s\")"%column)

        aggstr = ",".join(agglist)

        query = "SELECT %s FROM %s"%(aggstr, self.name)
        levels_tuple = self.ida_query(query, first_row_only = True)

        if len(levels_tuple) == 1:
            return levels_tuple[0]

        result = pd.Series(levels_tuple)
        result.index = columns
        return result


    #@timed
    @idadf_state
    def count_groupby(self, columns = None, count_only = False, having = None):
        """
        Count the occurence of the values of a column or group of columns
        """
        # TODO: Document, test
        if columns is not None:
            if isinstance(columns, six.string_types):
                columns = [columns]
            for column in columns:
                message = ''
                if column not in self.columns:
                    message += "Column %s does not belong to current idadataframe. \n"
                if message:
                    raise ValueError(message)
        else:
            columns = list(self.columns)
        if having:
            if not isinstance(having, Number):
                raise TypeError("having argument should be a number")

        select_columnstr = "\"" + "\",\"".join(columns) + "\", COUNT(*)"
        if count_only:
            select_columnstr = "COUNT(*)"
        groupby_columnstr = "\"" + "\",\"".join(columns) + "\""
        if having:
            groupby_columnstr = groupby_columnstr + " HAVING count >= %s"%having
        data = self.ida_query("SELECT %s AS count FROM %s GROUP BY %s"%(select_columnstr,self.name,groupby_columnstr))

        data.columns = columns + ["count"]
        return data

    def mean_freq_of_instance(self, columns = None):
        """
        Return the average occurence of the values of a column or group of columns
        """
        # TODO: Document, test
        if columns is not None:
            if isinstance(columns, six.string_types):
                columns = [columns]
            for column in columns:
                message = ''
                if column not in self.columns:
                    message += "Column %s does not belong to current idadataframe. \n"
                if message:
                    raise ValueError(message)
        else:
            columns = list(self.columns)

        select_columnstr = "COUNT(*)"
        groupby_columnstr = "\"" + "\",\"".join(columns) + "\""

        subquery = "SELECT %s AS count FROM %s GROUP BY %s"%(select_columnstr,self.name,groupby_columnstr)
        data = self.ida_scalar_query("SELECT AVG(DISTINCT count) FROM (%s)"%subquery)

        return int(data)

#    @timed
    @idadf_state
    def unique(self, column):
        """
        Return the unique values of a column
        """
        # TODO: Document, test
        if column in self._unique:
            return self._unique[column]

        #name = self.internal_state.current_state

        if not isinstance(column, six.string_types):
            raise TypeError("column argument not of string type")

        if column not in self.columns:
            # idadf.name somewhat false in case of modification
            raise ValueError("Undefined column \"%s\" in table %s"%(column, self._name))

        result = self.ida_query("SELECT DISTINCT \"%s\" FROM %s"%(column, self.name))

        self._unique[column] = result
        return result

    # TODO: to implement
    @timed
    @idadf_state
    def info(self, buf=None):
        """Some information about current IdaDataFrame. NOTIMPLEMENTED"""
        # There is a lot more
        from pandas.core.format import _put_lines

        if buf is None:  # pragma: no cover
            buf = sys.stdout

        lines = []

        lines.append(str(type(self)))
        lines.append(self.index.summary())

        if len(self.columns) == 0:
            lines.append('Empty %s' % type(self).__name__)
            _put_lines(buf, lines)
            return

###############################################################################
### Descriptive statistics
###############################################################################

    # TODO: to implement for categorical attributes
    @timed
    @idadf_state
    def describe(self, percentiles=[0.25, 0.50, 0.75]):
        """
        A basic statistical summary about current IdaDataFrame. If at least one
        numerical column exists, the summary includes:

            * The count of non-missing values for each numerical column.
            * The mean for each numerical column.
            * The standart deviation for each numerical column.
            * The minimum and maximum for each numerical column.
            * A list of percentiles set by the user (default : the quartiles).

        Parameters
        ----------
        idadf : IdaDataFrame
        percentiles : Float or list of floats, default: [0.25, 0.50, 0.75].
            percentiles to be computed on numerical columns.
            All values in percentiles must be > 0  and < 1.

        Returns
        -------
        summary: DataFrame, where
            * Index is the name of the computed values.
            * Columns are either numerical or categorical columns of self.

        """
        from ibmdbpy.statistics import describe
        return describe(idadf=self, percentiles=percentiles)

    @timed
    @idadf_state
    def cov(self):
        """
        Compute the covariance matrix, composed of covariance coefficients
        between all pairs of columns in self.
        It must have at least two numeric columns.

        Returns
        -------
        covariance matrix: DataFrame
            The axes are the columns of self and the values are the covariance
            coefficients.
        """
        from ibmdbpy.statistics import cov
        return cov(idadf=self)

    @timed
    @idadf_state
    def corr(self, method="pearson", features=None, ignore_indexer=True):
        """
        Compute the correlation matrix, composed of correlation coefficients
        between all pairs of columns in self.
        It must have at least two numeric columns.

        Parameters
        ----------
        method : str, default: pearson
            Method to be used to compute the correlation. By default, compute
            the pearson correlation coefficient. The Spearman rank correlation
            is also available. Admissible values are: "pearson", "spearman".

        Returns
        -------
        correlation matrix: DataFrame
            The axes are the columns of self and the values are the correlation
            coefficients.

        Notes
        -----
        For the Spearman rank correlation, the ordinal rank of columns is
        computed. For performance reasons this is easier to compute than the
        fractional rank traditionally computed for the Spearman rank
        correlation method. This strategy has the property that the sum of the
        ranking numbers is the same as under ordinal ranking. We then apply
        the pearson correlation coefficient method to these ranks.
        """
        from ibmdbpy.statistics import corr
        return corr(idadf=self, features=features, ignore_indexer=ignore_indexer)

    # TODO: to implement
    @timed
    @idadf_state
    def corrwith(self, other):
        """
        Compute the correlation matrix, composed of correlation coefficients
        between the columns of self and the columns of another IdaDataFrame.

        Parameters
        ----------
        other : DataFrame

        Returns
        -------
        correlation matrix: DataFrame
            The columns are the columns of self and the index the columns
            of other. The values are the covariance coefficients.
        """
        raise NotImplementedError("TODO")

    # TODO: to implement
    @timed
    @idadf_state
    def mode(self):
        """
        Compute the most common value for each non numeric column self. NOTIMPLEMENTED

        Returns
        -------
        mode: Series
        """
        raise NotImplementedError("TODO")

    @timed
    @idadf_state
    def quantile(self, q=0.5):
        """
        Compute row wise quantiles for each numeric column.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            0 <= q <= 1, the quantile(s) to compute

        Returns
        -------
        quantiles: Series or DataFrame
            If q is an array, the function returns a DataFrame in which the
            index is q. The columns are the columns of sel, and the values are
            the quantiles. If q is a float, a Series is returned where the
            index is the columns of self and the values are the quantiles.

        """
        from ibmdbpy.statistics import quantile
        return quantile(idadf=self, q=q)

    @timed
    @idadf_state
    def mad(self):
        """
        Compute the mean absolute distance for all numeric columns of self.
        It must have at least two numeric columns.

        Returns
        -------
        mad: Series
            The index consists of the columns of self and the values are the mean absolute distance.
        """
        from ibmdbpy.statistics import mad
        return mad(idadf=self)

    @timed
    @idadf_state
    def min(self):
        """
        Compute the minimum value for all numerics column of self.

        Returns
        -------
        min: Series
            The index consists of the columns of self and the values are the minimum.
        """
        from ibmdbpy.statistics import ida_min
        return ida_min(idadf=self)

    @timed
    @idadf_state
    def max(self):
        """
        Compute the maximum value over for all numeric columns of self.

        Returns
        -------
        max: Series.
            The index consists of the columns of self and the values are the maximum.
        """
        from ibmdbpy.statistics import ida_max
        return ida_max(idadf=self)

    @timed
    @idadf_state
    def count(self):
        """
        Compute the count of non-missing values for all columns of self.

        Returns
        -------
        count: Series.
            The index consists of the columns of self and the values are the number of non-missing values.
        """
        from ibmdbpy.statistics import count
        return count(idadf=self)

    @timed
    @idadf_state
    def count_distinct(self):
        # deprecated, use levels instead
        """
        Compute the count of distinct values for all numeric columns of self.

        Returns
        -------
        disctinct count: Series
            The index consists of the columns of self and values are the number of distinct values.
        """
        from ibmdbpy.statistics import count_distinct
        return count_distinct(idadf=self)

    @timed
    @idadf_state
    def std(self):
        """
        Compute the standard deviation for all numeric columns of self.

        Returns
        -------
        std: Series
            The index consists of the columns of self and the values are the standard deviation.
        """
        from ibmdbpy.statistics import std
        return std(idadf=self)

    @timed
    @idadf_state
    def within_class_var(self, target, features = None, ignore_indexer=True):
        if features is None:
            numerical_columns = self._get_numerical_columns()
            features = [x for x in numerical_columns if x != target]
        else:
            if isinstance(features, six.string_types):
                features = [features]

        if ignore_indexer is True:
            if self.indexer:
                if self.indexer in features:
                    features.remove(self.indexer)

        result = pd.Series()

        #C = self.levels(target)
        N = len(self)

        if len(features) < 5:

            avglist = ["AVG(\"%s\") as \"average%s\""%(feature, index) for index, feature in enumerate(features)]
            sumlist = ["SUM(CAST(POWER(\"%s\" - \"average%s\", 2) as DOUBLE))"%(feature, index) for index, feature in enumerate(features)]

            avgstr = ", ".join(avglist)
            sumstr = ", ".join(sumlist)

            subquery = "(SELECT \"%s\", %s FROM %s GROUP BY \"%s\") AS B"%(target, avgstr, self.name, target)
            condition = "ON A.\"%s\" = B.\"%s\""%(target, target)
            groupby = "GROUP BY A.\"%s\""%target
            query = "SELECT %s FROM %s AS A INNER JOIN %s %s %s"%(sumstr, self.name, subquery, condition, groupby)

            classvar = self.ida_query(query)
            if len(features) == 1:
                classvar = pd.DataFrame(classvar)

            C = len(classvar)
            if N == C:
                N += 1


            classvar.columns = pd.Index(features)
            result = pd.Series()
            for attr in classvar:
                result[attr] = classvar[attr].sum()/(N -C)

        else:
            chunkgen = chunklist(features, 5)
            result = pd.Series()
            for chunk in chunkgen:
                avglist = ["AVG(\"%s\") as \"average%s\""%(feature, index) for index, feature in enumerate(chunk)]
                sumlist = ["SUM(CAST(POWER(\"%s\" - \"average%s\", 2) as DOUBLE))"%(feature, index) for index, feature in enumerate(chunk)]

                avgstr = ", ".join(avglist)
                sumstr = ", ".join(sumlist)

                subquery = "(SELECT \"%s\", %s FROM %s GROUP BY \"%s\") AS B"%(target, avgstr, self.name, target)
                condition = "ON A.\"%s\" = B.\"%s\""%(target, target)
                groupby = "GROUP BY A.\"%s\""%target
                query = "SELECT %s FROM %s AS A INNER JOIN %s %s %s"%(sumstr, self.name, subquery, condition, groupby)

                classvar = self.ida_query(query)

                if len(chunk) == 1:
                    classvar = pd.DataFrame(classvar)

                C = len(classvar)
                if N == C:
                    N += 1

                classvar.columns = pd.Index(chunk)

                for attr in classvar:
                    result[attr] = classvar[attr].sum()/(N -C)

        return result

    @timed
    @idadf_state
    def within_class_std(self, target, features = None, ignore_indexer= True):
        return np.sqrt(self.within_class_var(target, features, ignore_indexer))

    @timed
    @idadf_state
    def var(self):
        """
        Compute the variance for all numeric columns of self.

        Returns
        -------
        var: Series
            The index consists of the columns of self and the values are the variance.
        """
        from ibmdbpy.statistics import var
        return var(idadf=self)

    @timed
    @idadf_state
    def mean(self):
        """
        Compute the mean for each numeric columns of self.

        Returns
        -------
        mean: Series
            The index consists of the columns of self and the values are the mean.
        """
        from ibmdbpy.statistics import mean
        return mean(idadf=self)

    @timed
    @idadf_state
    def mean_groupby(self, groupby, features = None):
        if features is None:
            features = [x for x in self.columns if x != groupby]
        else:
            if isinstance(features, six.string_types):
                features = [features]

        avglist = ["CAST(AVG(\"%s\") as FLOAT)"%feature for feature in features]
        avgstr = ", ".join(avglist)

        query = "SELECT \"%s\", %s FROM %s GROUP BY \"%s\""%(groupby, avgstr, self.name, groupby)
        result = self.ida_query(query)
        result = result.pivot_table(index = result.columns[0])
        result.columns = pd.Index(features)
        return result


    @timed
    @idadf_state
    def sum(self):
        """
        Compute the sum of values for all numeric columns of self.

        Returns
        -------
        sum: Series
            The index consists of the columns of self and the values are the sum.
        """
        # Behave like having the option "numeric only" to true
        # TODO: Implement the option
        from ibmdbpy.statistics import ida_sum
        return ida_sum(idadf=self)

    @timed
    @idadf_state
    def median(self):
        """
        Compute the median for all numeric columns of self.

        Returns
        -------
        median: Series
            The index consists of the columns of self and the values are the median.
        """
        # Behave like having the option "numeric only" to true
        # TODO: Implement the option
        from ibmdbpy.statistics import median
        return median(idadf=self)

    # Maybe not be possible
    @idadf_state
    def rank(self):
        """Compute the rank over all entries for all columns of self."""
        raise NotImplementedError("TODO")

    # TODO : cumsum, cummean, cummcountm cummax, cumprod

###############################################################################
### Save current IdaDataFrame to Db2 Warehouse as a table
###############################################################################

    # TODO: Should this function be in IdaDataBase ?
    # To my mind, it is more intuitive to let it here, but it is "destructive"
    @timed
    @idadf_state
    def save_as(self, tablename, clear_existing = False):
        """
        Save self as a table name in the remote database with the name
        tablename. This function might erase an existing table if tablename
        already exists and clear_existing is True.

        """
        # TODO: to test !
        if tablename == self.tablename:
            if clear_existing is False:
                raise ValueError("Cannot overwrite current IdaDataFrame if "+
                                " clear_existing option set to False.")
            message = "Table %s already exists."%(tablename)
            warnings.warn(message, UserWarning)
            question = "Are you sure that you want to overwrite %s"%(tablename)
            display_yes = ibmdbpy.utils.query_yes_no(question)
            if not display_yes:
                return
            tempname = self._idadb._get_valid_tablename()
            self._prepare_and_execute("CREATE TABLE %s AS (SELECT * FROM %s) WITH DATA"%(tempname, tablename))
            try:
                self._idadb.drop_table(tablename)
            except:
                self._idadb.drop_view(tablename)

            newidadf = IdaDataFrame(self._idadb, tempname)
            self._idadb.rename(newidadf, tablename)
            self = newidadf

        if self._idadb.exists_table_or_view(tablename):
            if clear_existing:
                message = "Table %s already exists."%(tablename)
                warnings.warn(message, UserWarning)
                question = "Are you sure that you want to overwrite %s"%(tablename)
                display_yes = ibmdbpy.utils.query_yes_no(question)
                if not display_yes:
                    return
                try:
                    self._idadb.drop_table(tablename)
                except:
                    self._idadb.drop_view(tablename)
            else:
                raise NameError(("%s already exists, choose a different name "+
                                "or use clear_existing option.")%tablename)

        name = self.internal_state.current_state

        self._prepare_and_execute("CREATE TABLE %s AS (SELECT * FROM %s) WITH DATA"%(tablename, name))

        # Reset the cache
        self._idadb._reset_attributes("cache_show_tables")

###############################################################################
### Import as DataFrame
###############################################################################

    @timed
    @idadf_state
    def as_dataframe(self):
        """
        Download and return an in-memory representation of the dataset as
        a Pandas DataFrame.

        Returns
        -------
        DataFrame
            Columns and records are the same as in self.


        Examples
        --------
        >>> iris = ida_iris.as_dataframe()
        >>> iris.head()
           sepal_length  sepal_width  petal_length  petal_width species
        0           5.1          3.5           1.4          0.2  setosa
        1           4.9          3.0           1.4          0.2  setosa
        2           4.7          3.2           1.3          0.2  setosa
        3           4.6          3.1           1.5          0.2  setosa
        4           5.0          3.6           1.4          0.2  setosa
        """
        if os.environ['VERBOSE'] == 'True':
            # We use an empirical estimation
            # Experimental results :
            # 1M row, 12 columns => 550 secs
            # 3.77M, 12 columns => 2256 secs
            raw_estimation = ((self.shape[0]*self.shape[1]) / 4000000) * 3
            if raw_estimation < 0.01:
                # There is a small offset, which is negligeable for higher measurements
                estimation = str(raw_estimation*60 + 0.717) + " seconds."
            elif raw_estimation > 60:
                estimation = str(raw_estimation/60) + " hours."
            else:
                estimation = "%s minutes."%raw_estimation
            print("Your IdaDataFrame has %s rows and %s columns." % (self.shape[0], self.shape[1]))
            print("Estimated download time : ~ " + estimation)
            if raw_estimation > 30:
                message = ("Estimated Download time is greater than" +
                           " %s minutes.")%raw_estimation
                warnings.warn(message, UserWarning)
                question = "Do you want to download the dataset?"
                display_yes = ibmdbpy.utils.query_yes_no(question)
                if not display_yes:
                    return

        data = self.ida_query(self.internal_state.get_state())
        data.columns = self.columns
        data.name = self.tablename
        # Handle datatypes
#        data = ibmdbpy.utils._convert_dtypes(self, data)
        return data

###############################################################################
### Connection Management
###############################################################################

# Connection is a bit different, this is made to allow user to work on several
# IdaDataFrame and close them, without closing the connection. This is why
# function "close" is overwritten and a function reopen is provided

    # Not sure commit and rollback should be here ?

    def commit(self):
        """
        Commit operations in the database.

        Notes
        -----

        All changes that are made in the database after the last commit,
        including those in the child IdaDataFrames, are commited.

        If the environment variable ‘VERBOSE’ is set to True, the commit
        operations are notified in the console.
        """
        self._idadb.commit()

    def rollback(self):
        """
        Rollback operations in the database.

        Notes
        -----

        All changes that are made in the database after the last commit,
        including those in the child IdaDataFrames, are discarded.
        """
        self._idadb.rollback()


###############################################################################
### Private functions
###############################################################################

    def _clone(self):
        """
        Clone the actual object.
        """
        newida = IdaDataFrame(self._idadb, self._name, self.indexer)
        newida.columns = self.columns
        newida.dtypes = self.dtypes     # avoid recomputing it
        # otherwise risk of infinite loop between
        # idadf.columns and internalstate.columndict

        # This is not possible to use deepcopy on an IdaDataFrame object
        # because the reference to the parents IdaDataBase with the connection
        # object is not pickleable. As a consequence, we create a new
        # IdaDataFrame and copy all the relevant attributes

        newida.internal_state.name = deepcopy(self.internal_state.name)
        newida.internal_state.ascending = deepcopy(self.internal_state.ascending)
        #newida.internal_state.views = deepcopy(self.internal_state.views)
        newida.internal_state._views = deepcopy(self.internal_state._views)
        newida.internal_state._cumulative = deepcopy(self.internal_state._cumulative)
        newida.internal_state.order = deepcopy(self.internal_state.order)
        newida.internal_state.columndict = deepcopy(self.internal_state.columndict)
        return newida

    def _clone_as_serie(self, column):
        """
        Clone the actual object as an IdaSeries and select one of its columns.
        """
        newida = ibmdbpy.IdaSeries(idadb = self._idadb, tablename = self._name,
                                   indexer = self.indexer, column = column)

        newida.internal_state.name = deepcopy(self.internal_state.name)
        newida.internal_state.ascending = deepcopy(self.internal_state.ascending)
        #newida.internal_state.views = deepcopy(self.internal_state.views)
        newida.internal_state._views = deepcopy(self.internal_state._views)
        newida.internal_state._cumulative = deepcopy(self.internal_state._cumulative)
        newida.internal_state.order = deepcopy(self.internal_state.order)
        newida.internal_state.columndict = deepcopy(self.internal_state.columndict)
        return newida

    def _get_type(self):
        """
        Type of the IdaDataFrame : "Table", "View" or "Unknown".
        """
        if self._idadb.is_table(self._name):
            return "Table"
        elif self._idadb.is_view(self._name):
            return "View"
        else:
            return "Unkown"

    def _get_columns(self):
        """
        Index containing a list of the columns in self.
        """
        tablename = self.internal_state.current_state
        if self._idadb._con_type == 'odbc':
            if '.' in tablename:
                schema = tablename.split('.')[-2]
                tablename = tablename.split('.')[-1]
            else:
                schema = self._idadb.current_schema
            columns = self._idadb._con.cursor().columns(table=tablename, schema=schema)
            columnlist = [column[3] for column in columns]
            return Index(columnlist)
        elif self._idadb._con_type == 'jdbc':
            cursor = self._idadb._con.cursor()
            cursor.execute("SELECT * FROM %s"%tablename)
            columnlist = [column[0] for column in cursor.description]
            return Index(columnlist)

    def _get_all_columns_in_table(self):
        """Get all columns that exists in the physical table."""
        return self._get_columns()

    # TODO: To deprecate
    def _get_index(self, force=False):
        """
        Index containing a list of the row names in self.
        """

        rows = self.shape[0]

        return RangeIndex(0, rows, 1)

    def _get_shape(self):
        """
        Tuple containing the number of rows and the number of columns in self.
        """
        name = self.internal_state.current_state

        nrow = self.ida_scalar_query("SELECT CAST(COUNT(*) AS INTEGER) FROM %s"%name)
        ncol = len(self.columns)
        return (nrow, ncol)

    def _get_columns_dtypes(self):
        """
        DataFrame containing the column names and database types in self.
        """
        name = self.internal_state.current_state

        # In case the name is composed the following way : SCHEMA.TABLENAME
        # We need to separate it to keep only the TABLENAME for this query.
        if '.' in name :
            name = name.split('.')[-1]

        if name.find("TEMP_VIEW_") == 0:
            #When the column names are going to be retrieved from a temporary
            #view that was created with the definition of the current state of
            #the IdaDataFrame, the schema name cannot be assumed as the same of
            #the IdaDataFrame. Also mind that the name of the temporary view
            #is thought to be random enough to avoid collisions
            data = self.ida_query(("SELECT COLNAME, TYPENAME FROM SYSCAT.COLUMNS "+
                               "WHERE TABNAME=\'%s\' "+
                               "ORDER BY COLNO")%(name))
        else:
           data = self.ida_query(("SELECT COLNAME, TYPENAME FROM SYSCAT.COLUMNS "+
                               "WHERE TABNAME=\'%s\' AND TABSCHEMA=\'%s\' "+
                               "ORDER BY COLNO")%(name, self.schema))

        # Workaround for some ODBC version which does not get the entire
        # string of the column name in the cursor descriptor.
        # This is hardcoded, so be careful
        data.columns = ["COLNAME", "TYPENAME"]
        data.columns = [x.upper() for x in data.columns]
        data.set_index(keys='COLNAME', inplace=True)
        data.index.name = None
        return data

    def _reset_attributes(self, attributes):
        """
        Delete an attribute of self to force its refreshing at the next call.
        """
        if isinstance(attributes, six.string_types):
            attributes = [attributes]

        # Special case : resetting columns
        if "columns" in attributes:
             try:
                 del self.internal_state.columndict
             except:
                 pass

        ibmdbpy.utils._reset_attributes(self, attributes)


###############################################################################
### DB2 Warehouse to pandas type mapping
###############################################################################

    def _table_def(self, factor_threshold=None):
        """
        Classify columns in the idaDataFrame into 4 classes: CATEGORICAL,
        STRING, NUMERIC or NONE. Use the database data type and a
        user-threshold “factor_threshold”:

            * CATEGORICAL columns that have a number of distinct values that is greater
              than the factor_threshold should be considered a STRING.
            * NUMERIC columns that have a number of distinct values that is smaller or equal
              to the factor_threshold should be considered CATEGORICAL.

        Returns
        -------
        DataFrame

            * Index is the columns of self.
            * Column "FACTORS" contains the number of distinct values.
            * Column "VALTYPE" contains the resulting class.

        Examples
        --------
        >>> ida_iris._table_def()
                     TYPENAME FACTORS      VALTYPE
        sepal_length   DOUBLE      35      NUMERIC
        sepal_width    DOUBLE      23      NUMERIC
        petal_length   DOUBLE      43      NUMERIC
        petal_width    DOUBLE      22      NUMERIC
        species       VARCHAR       3  CATEGORICAL
        """
        # We don't want to change the value of the attribute
        data = deepcopy(self.dtypes)

        def _valtype_from_dbtype(tup):
            """
            Decides if a column should be considered categorical or numerical
            """
            categorical_attributes = ['VARCHAR', 'CHARACTER', 'VARGRAPHIC',
                                      'GRAPHIC', 'CLOB']
            numerical_attributes = ['SMALLINT', 'INTEGER', 'BIGINT', 'REAL',
                                    'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC']
            if tup[0] in categorical_attributes:
                if factor_threshold is None:
                    return "CATEGORICAL"
                elif tup[1] <= factor_threshold:
                    return "CATEGORICAL"
                else:
                    return "STRING"
            elif tup[0] in numerical_attributes:
                if factor_threshold is None:
                    return "NUMERIC"
                elif tup[1] > factor_threshold:
                    return "NUMERIC"
                else:
                    return "CATEGORICAL"
            else:
                return "NONE"

        data['FACTORS'] = ibmdbpy.statistics._count_level(self, data.index.values)
        data['VALTYPE'] = [_valtype_from_dbtype(x) for x in
                           data[['TYPENAME', 'FACTORS']].to_records(index=0)]
        return data

    def _get_numerical_columns(self):
        """
        Get the columns of self that are considered as numerical. Their data
        type in the database determines whether these columns are numerical.
        The following data types are considered numerical:

            SMALLINT, INTEGER, BIGINT, REAL, DOUBLE, FLOAT, DECIMAL, NUMERIC.

        Returns
        -------
        list
            List of numerical column names.

        Examples
        --------
        >>> ida_iris._get_numerical_columns()
        ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        """
        num = ['SMALLINT', 'INTEGER', 'BIGINT', 'REAL',
                            'DOUBLE', 'FLOAT', 'DECIMAL', 'NUMERIC']
        return list(self.dtypes.loc[self.dtypes['TYPENAME'].isin(num)].index)

    def _get_categorical_columns(self):
        """
        Get the columns of self that are considered as categorical. Their data
        type in the database determines whether these columns are categorical.
        The following data types are considered categorical:

            VARCHAR,CHARACTER, VARGRAPHIC, GRAPHIC, CLOB.

        Returns
        -------
        list
            List of categorical column names.

        Examples
        --------
        >>> ida_iris._get_categorical_columns()
        ['species']
        """
        cat = ['VARCHAR', 'CHARACTER', 'VARGRAPHIC', 'GRAPHIC', 'CLOB']
        return list(self.dtypes.loc[self.dtypes['TYPENAME'].isin(cat)].index)

    def _prepare_and_execute(self, query, autocommit = True, silent = False):
        """
        Prepare and execute a query.

        Parameters
        ----------
        query : str
            Query to be executed.
        autocommit : bool
             If set to true, the autocommit function is available.
        """
        return self._idadb._prepare_and_execute(query, autocommit, silent)

    def _autocommit(self):
        """
        Autocommit the connection. If the environment variable ‘AUTOCOMMIT’ is
        set to True, the function commits the changes.

        Notes
        -----

        If you commit, all changes that are made in the database after the last
        commit, including those in the child IdaDataFrames, are commited.

        If the environment variable ‘VERBOSE’ is set to True, the autocommit
        operations are notified in the console.
        """
        self._idadb._autocommit()

    def _combine_check(self, other):
        """
        Check if self and other refer to the same table and if all columns in
        self and other are numeric. This sanity check is used before performing
        aggregation operations between IdaDataFrame/IdaSeries.
        """
        def check_numeric_columns(idaobject):
            not_valid = []
            numeric_columns = idaobject._get_numerical_columns()
            for column in idaobject.columns:
                if column not in numeric_columns:
                    not_valid.append(column)
            if not_valid:
                raise TypeError("Arithmetic operation are not defined for %s"%not_valid)

        if isinstance(other, IdaDataFrame) | isinstance(other, ibmdbpy.IdaSeries):
            if self._name != other._name:
                raise IdaDataFrameError("It is not possible to aggregate columns using columns of a different table.")

        if not(isinstance(other, IdaDataFrame) | isinstance(other, ibmdbpy.IdaSeries) | isinstance(other, Number)):
            if other is not None:
                raise TypeError("Aggregation makes only sense with numbers, "+
                                "or IdaDataFrames refering to the same table.")

        check_numeric_columns(self)
        if isinstance(other, ibmdbpy.IdaSeries)|isinstance(other, IdaDataFrame):
            check_numeric_columns(other)

    
    def delete_na(self, columns, logic="any", inplace=False):
        """
        Filter rows containing NULL values. Can be done in a destructive way (rows are deleted in the physical table)
        or in a non destructive way (a new IdaDataFrame is defined, original table and IdaDataFrame are preserved). 
        
        Parameters
        -----------
        columns : list of strings
            list of eligible column names
        logic : str, optional
            "any" by default. 
            If logic is set to "any" then all rows which contain a NaN value in any 
            (id est at least one) of the cited columns is deleted, this is a union condition;
            if logic is set to "all", then only rows containing null values in all the fields are deleted, this is an intersection condition.         
        inplace : bool, optional 
            False by default. 
            If True, then the underlying table is physically modified,
            if False, then the original objects remain unaffected, a copy of the IdaDataFrame is made and modified. 
        
        Returns
        --------
        If inplace is True, the original table is modified and the IdaDataFrame will be modified accordingly. No return.
        If inplace is False, a new IdaDataFrame is returned, the original table is not affected. This new IdaDataFrame points to a 
        new table which has been created according to the user defined criterions on columns.
        
        Examples
        --------
        >>> idadf.delete_na(["COL_1", "COL_2"], inplace = True)
        >>> #each row of the original table is physically deleted if there is a NULL value in one of the listed columns
        >>> new_idadf = idadf.delete_na(["COL_1", "COL_2"], logic = 'all')
        >>> # a new IdaDataFrame is created, rows will be selected only if all the listed columns have a NULL value.
        """
        
        # Check if list columns is not empty
        if len(columns)<1:
            raise IdaDataFrameError("You must specify the columns as a list of eligible names")
        # Check if column names are eligible
        for column_name in columns:
            if not isinstance(column_name, six.string_types):
                raise TypeError("%s is not of string type"%(column_name))
            if column_name not in self.columns:
                raise ValueError("%s refers to a columns that doesn't exists in self"%(column_name))
        
        idadb = self._idadb
        
        if inplace == True:
            # Explain
            print("The table %s will be physically modified." %self.tablename)
            print("Any IdaDataFrame pointing this table might be modified accordingly.")
            # Write DELETE query on original table
            tablename = self.tablename
            query = 'DELETE FROM %s WHERE "%s" IS NULL'%(tablename, columns[0])
            
            if len(columns)>1:
                if logic == "any":
                    for col in columns[1:]:
                        query = query + ' OR "%s" IS NULL'%col
                if logic == "all":
                    for col in columns[1:]:
                        query = query + ' AND "%s" IS NULL'%col                        
            idadb.ida_query(query)

        else:            
            # SELECT statement on the original table to create a view
            tablename0 = self.tablename 
            query = 'SELECT * FROM %s WHERE "%s" IS NOT NULL'%(tablename0, columns[0])
            
            if len(columns)>1:
                if logic == "any":
                    for col in columns[1:]:
                        query = query + ' AND "%s" IS NOT NULL'%col
                if logic == "all":
                    for col in columns[1:]:
                        query = query + ' OR "%s" IS NOT NULL'%col                                               
            # Create view with this select statement
            viewname = idadb._get_valid_tablename(prefix="VIEW_")
            self._prepare_and_execute("CREATE VIEW " + viewname + " AS "+ query)
            # Initiate the modified table under a random name
            tablename = idadb._get_valid_tablename(prefix="DATA_FRAME_")
            idadb._prepare_and_execute("CREATE TABLE %s AS (SELECT * FROM %s) WITH DATA"%(tablename,viewname))
            print('A new table with filtered rows is available under the name %s.' %tablename)
            print('The newly created IdaDataFrame refers to this new table.')
            # Drop the view 
            idadb.drop_view(viewname)            
            # Define a new IdaDataFrame pointing to the new table
            idadf = IdaDataFrame(idadb, tablename, indexer = self.indexer)
            return idadf
    
            


            
        
