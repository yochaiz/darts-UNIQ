from os import path, makedirs
from datetime import datetime
from io import BytesIO
from base64 import b64encode
from urllib.parse import quote

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class HtmlLogger:
    timestampColumnName = 'Timestamp'

    def __init__(self, save_path, filename):
        self.save_path = save_path
        self.filename = filename
        self.fullPath = '{}/{}.html'.format(save_path, filename)

        if not path.exists(save_path):
            makedirs(save_path)

        if path.exists(self.fullPath):
            with open(self.fullPath, 'r') as f:
                content = f.read()
            # remove close tags in order to allow writing to data table
            for v in ['</body>', '</html>', '</table>']:
                idx = content.rfind(v)
                # remove tag from string
                if idx >= 0:
                    content = content[:idx] + content[idx + len(v):]

            self.head = content
            # script already in self.head now, therefore no need it again
            self.script = ''
        else:
            self.head = '<!DOCTYPE html><html><head><style>' \
                        'table { font-family: gisha; border-collapse: collapse;}' \
                        'td, th { border: 1px solid #dddddd; text-align: center; padding: 8px; white-space:pre;}' \
                        '.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; border: none; text-align: left; outline: none; font-size: 15px; }' \
                        '.active, .collapsible:hover { background-color: #555; }' \
                        '.content { max-height: 0; overflow: hidden; transition: max-height 0.2s ease-out;}' \
                        '</style></head>' \
                        '<body>'
            # init collapse script
            self.script = '<script> var coll = document.getElementsByClassName("collapsible"); var i; for (i = 0; i < coll.length; i++) { coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.maxHeight){ content.style.maxHeight = null; } else { content.style.maxHeight = content.scrollHeight + "px"; } }); } </script>'

        self.end = '</body></html>'
        self.infoTables = ''
        self.dataTable = ''

    # converts dictionary to rows with nElementPerRow (k,v) elements at most in each row
    @staticmethod
    def dictToRows(dict, nElementPerRow):
        rows = []
        row = []
        counter = 0
        # sort elements by keys name
        for k in sorted(dict.keys()):
            v = dict[k]
            row.append(k)
            row.append(v)
            counter += 1

            if counter == nElementPerRow:
                rows.append(row)
                row = []
                counter = 0

        # add last elements
        if len(row) > 0:
            rows.append(row)

        return rows

    def __writeToFile(self):
        # init elements write order to file
        writeOrder = [self.head, self.infoTables, self.script, self.dataTable, '</table>', self.end]
        # write elements
        with open(self.fullPath, 'w') as f:
            for elem in writeOrder:
                if elem is not '':
                    f.write(elem)

    def __addRow(self, row):
        res = '<tr>'
        for v in row:
            # check maybe we have a sub-table
            if (type(v) is list) and (len(v) > 0) and (type(v[0]) is list):
                v = self.__createTableFromRows(v)
            # add element or sub-table to current table
            res += '<td> {} </td>'.format(v)
        res += '</tr>'

        return res

    # recursive function that supports sub-tables
    def __createTableFromRows(self, rows):
        res = '<table>'
        # create rows
        for row in rows:
            res += self.__addRow(row)
        # close table
        res += '</table>'
        return res

    # title - a string for table title
    # rows - array of rows. each row is array of values.
    def addInfoTable(self, title, rows):
        res = '<button class="collapsible"> {} </button>'.format(title)
        res += '<div class="content">'
        res += self.__createTableFromRows(rows)
        res += '</div>'
        # add table to body
        self.infoTables += res
        # create gap for next table
        self.infoTables += '<h2></h2>'
        # write to file
        self.__writeToFile()

    def addRowToInfoTable(self, title, row):
        valuesToFind = [title, '</table>']
        idx = 0
        # walk through the string to the desired position
        for v in valuesToFind:
            if idx >= 0:
                idx = self.head.find(v, idx)

        if idx >= 0:
            # insert new row in desired position
            self.head = self.head[:idx] + self.__addRow(row) + self.head[idx:]
            # write to file
            self.__writeToFile()

    @staticmethod
    def __addColumnsRowToTable(cols):
        res = '<tr bgcolor="gray">'
        for c in cols:
            res += '<td> {} </td>'.format(c)
        res += '</tr>'
        # returns columns row
        return res

    def addColumnsRowToDataTable(self, writeToFile=False):
        self.dataTable += self.__addColumnsRowToTable(self.dataTableCols)
        # write to file
        if writeToFile:
            self.__writeToFile()

    def updateDataTableCols(self, dataTableCols):
        # save copy of columns names
        self.dataTableCols = dataTableCols.copy()
        # add timestamp to columns
        self.dataTableCols.insert(0, self.timestampColumnName)
        # save table number of columns
        self.nColsDataTable = len(self.dataTableCols)

    @staticmethod
    def __addTitleRow(title, nCols):
        return '<tr><th colspan={} bgcolor="gray"> {} </th></tr>'.format(nCols, title)

    def createDataTable(self, title, columns):
        res = ''
        # check if we need to close last data table in page, before starting a new one
        if len(self.dataTable) > 0:
            res += '</table><h2></h2>'

        res += '<table>'
        # update data table columns
        self.updateDataTableCols(columns)
        # create title row
        res += self.__addTitleRow(title, self.nColsDataTable)
        # add table to body
        self.dataTable += res
        # add columns row
        self.addColumnsRowToDataTable()
        # write to file
        self.__writeToFile()

    def getTimeStr(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # values is a dictionary
    def addDataRow(self, values, trType='<tr>', writeFile=True):
        res = trType
        # add timestamp to values
        values[self.timestampColumnName] = self.getTimeStr()
        # build table row, iterate over dictionary
        for c in self.dataTableCols:
            res += '<td>'
            if c in values:
                if isinstance(values[c], list):
                    res += self.__createTableFromRows(values[c])
                else:
                    res += '{}'.format(values[c])
            res += '</td>'
        # close row
        res += '</tr>'
        # add data to dataTable
        self.dataTable += res
        if writeFile:
            # write to file
            self.__writeToFile()

    # add data summary to data table
    # values is a dictionary
    def addSummaryDataRow(self, values):
        self.addDataRow(values, trType='<tr bgcolor="#27AE60">')

    def addInfoToDataTable(self, line, color='lightblue'):
        res = '<tr>'
        res += '<td> {} </td>'.format(self.getTimeStr())
        res += '<td colspan={} bgcolor="{}"> {} </td>'.format(self.nColsDataTable - 1, color, line)
        res += '</tr>'
        # add table to body
        self.dataTable += res
        # write to file
        self.__writeToFile()

    def plot(self, **kwargs):
        # data is a list, where each element is [x , y , 'bo' (i.e. pts style]
        data = kwargs.get('data')

        if not data:
            return

        fig, ax = plt.subplots(nrows=1, ncols=1)
        for x, y, style in data:
            ax.plot(x, y, style)

        # init properties we might want to handle
        properties = [('xticks', ax.set_xticks), ('yticks', ax.set_yticks), ('size', fig.set_size_inches),
                      ('xlabel', ax.set_xlabel), ('ylabel', ax.set_ylabel), ('title', ax.set_title)]

        for key, func in properties:
            if key in kwargs:
                func(kwargs[key])

        # set title
        infoTableTitle = kwargs.get('title', 'Plot')

        # convert fig to base64
        canvas = FigureCanvas(fig)
        png_output = BytesIO()
        canvas.print_png(png_output)
        img = b64encode(png_output.getvalue())
        img = '<img src="data:image/png;base64,{}">'.format(quote(img))

        self.addInfoTable(infoTableTitle, [[img]])
