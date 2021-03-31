from mock import MagicMock, Mock
import unittest
import sqlite3
from app import *
# http://www.voidspace.org.uk/python/mock/

class Database_Test_Case(unittest.TestCase):

    def test_sqlite_successfulConnect(self):
        sqlite3.connect = MagicMock(return_value='connection succeeded')

        dbc = DataBaseClass()
        sqlite3.connect.assert_called_with('reddit_comments.db')
        self.assertEqual(dbc.connection, 'connection succeeded')

    def test_sqlite_failConnect(self):
        sqlite3.connect = MagicMock(return_value='connection failed')

        dbc = DataBaseClass()
        sqlite3.connect.assert_called_with('reddit_comments.db')
        self.assertEqual(dbc.connection, 'connection failed')

    def test_sqlite3_connect_with_sideaffect(self):

        self._setup_mock_sqlite3_connect()

        dbc = DataBaseClass('good_connection_string')
        self.assertTrue(dbc.connection)
        sqlite3.connect.assert_called_with('good_connection_string')

        dbc = DataBaseClass('bad_connection_string')
        self.assertFalse(dbc.connection)
        sqlite3.connect.assert_called_with('bad_connection_string')

    def _setup_mock_sqlite3_connect(self):

        values = {'good_connection_string':True,
                  'bad_connection_string':False}

        def side_effect(arg):
            return values[arg]

        sqlite3.connect = Mock(side_effect=side_effect)

    def test_connDB(self):
        cursor = connDB()
        isCursor = True
        if cursor is None:
            isCursor = False

        self.assertEqual(isCursor, True)


class DataBaseClass():

    def __init__(self,connection_string='reddit_comments.db'):
        self.connection = sqlite3.connect(connection_string)

if __name__ == '__main__':
    unittest.main()
