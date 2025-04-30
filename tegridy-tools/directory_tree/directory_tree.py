#======================================================================================
#
# Directory Tree Python Module
#
# Partial directory_tree code as a stand-alone Python module
#
# Version 1.0
#
# Original source code courtesy of Rahul Bordoloi
# https://github.com/rahulbordoloi/Directory-Tree
#
# Original source code retrieved on 04/30/2025
# Original version 1.0.0 / Commit 921aa74
#
# Project Los Angeles
# Tegridy Code 2025
#
#======================================================================================
# Imports
#======================================================================================

from __future__ import annotations
from fnmatch import fnmatch
from os import stat
from pathlib import Path
from stat import FILE_ATTRIBUTE_HIDDEN
from os import getcwd
from pathlib import Path
from platform import system
from traceback import format_exc
from typing import Any, List, Union

#======================================================================================
# tree_calculator.py
#======================================================================================

# Class for Calculating Directory Tree Path
class DirectoryPath:
    """
    Python Utility Package that Displays out the Tree Structure of a Particular Directory.
    @author : rahulbordoloi
    """

    # Class Variables [Directions]
    displayNodePrefixMiddle: str = '├──'
    displayNodePrefixLast: str = '└──'
    displayParentPrefixMiddle: str = '    '
    displayParentPrefixLast: str = '│   '

    # Constructor
    def __init__(self, path: Path, parentPath: Union[DirectoryPath, None]=None, isLast: bool=False) -> None:

        # Instance Variables [Status of Parent-Node Files]
        self.path: Path = Path(path)
        self.parent: DirectoryPath = parentPath
        self.isLast: bool = isLast
        if self.parent:
            self.depth: int = self.parent.depth + 1
        else:
            self.depth: int = 0

    # Destructor
    def __del__(self) -> None:
        del self.path, self.parent, self.isLast, self.depth

    # Displaying Names of the Nodes [Parents / Inner Directories]
    @property
    def displayName(self) -> str:
        """
        Method to Display the Name of the Nodes [Parents / Inner Directories]
        """

        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    # Building the Tree [Directories - Nodes]
    @classmethod
    def buildTree(cls, root: Path, parent: Union[DirectoryPath, None]=None, isLast: bool=False,
                  maxDepth: float=float('inf'), showHidden: bool=False, ignoreList: List[str]=None,
                  onlyFiles: bool=False, onlyDirectories: bool=False, sortBy: int=0) -> str:
        """
        Method to Build the Tree Structure of the Directory
        :param root: Root Path of the Directory
        :param parent: Parent Directory Path
        :param isLast: Boolean Flag for Last Node
        :param maxDepth: Max Depth of the Tree
        :param showHidden: Boolean Flag for Displaying Hidden Files/Directories
        :param ignoreList: List of Files/Directories to Ignore
        :param onlyFiles: Boolean Flag for Displaying Only Files
        :param onlyDirectories: Boolean Flag for Displaying Only Directories
        :param sortBy: Sorting Order of the Files / Directory
        :return: String Representation of the Tree
        """

        # Resolving `Ignore List`
        if not ignoreList:
            ignoreList: List[str] = []

        # Generator Method to Generate Tree
        root: Path = Path(root)
        rootDirectoryDisplay: DirectoryPath = cls(
            path=root,
            parentPath=parent,
            isLast=isLast
        )
        yield rootDirectoryDisplay

        ## Taking out the List of Children [Nodes] Files/Directories
        children: List[Path] = sorted(
            list(entityPath for entityPath in root.iterdir()), key=lambda s: str(s).lower()
        )

        ## Checking for Hidden Entities Flag
        if not showHidden:
            children: List[Path] = [
                entityPath for entityPath in children if not cls._hiddenFilesFiltering_(entityPath)
            ]

        # Filter out Entities (Files and Directories) Specified in the `ignore_list`
        children: List[Path] = [
            entityPath for entityPath in children
            if not any(
                fnmatch(str(entityPath.relative_to(root)), entity) or fnmatch(entityPath.name, entity) for entity in ignoreList
            )
        ]

        # Filtering out based on `onlyFiles` and `onlyDirectories` Flags
        if onlyFiles:
            children: List[Path] = [
                entityPath for entityPath in children if entityPath.is_file()
            ]
        elif onlyDirectories:
            children: List[Path] = [
                entityPath for entityPath in children if entityPath.is_dir()
            ]
        if onlyFiles and onlyDirectories:
            raise AttributeError('Only One of `onlyFiles` and `onlyDirectories` Flags can be Set to `True`')

        # Sorting based on `sortBy` Flag [ 1 - Files First, 2 - Directories First ]
        if sortBy == 1:
            children.sort(key=lambda s: (s.is_dir(), str(s).lower()))
        elif sortBy == 2:
            children.sort(key=lambda s: (not s.is_dir(), str(s).lower()))
        else:
            children.sort(key=lambda s: str(s).lower())

        countNodes: int = 1
        for path in children:
            isLast: bool = countNodes == len(children)
            if path.is_dir() and rootDirectoryDisplay.depth + 1 < maxDepth:
                yield from cls.buildTree(
                    root=path,
                    parent=rootDirectoryDisplay,
                    isLast=isLast,
                    maxDepth=maxDepth,
                    showHidden=showHidden,
                    ignoreList=ignoreList,
                    onlyFiles=onlyFiles,
                    onlyDirectories=onlyDirectories,
                    sortBy=sortBy
                )
            else:
                yield cls(
                    path=path,
                    parentPath=rootDirectoryDisplay,
                    isLast=isLast
                )
            countNodes += 1

    # Check Condition for Hidden Entities [Files / Directories]
    @classmethod
    def _hiddenFilesFiltering_(cls, path: Path) -> bool:
        """
        Method to Check for Hidden Files / Directories
        :param path: Path of the File / Directory
        """

        try:
            return bool(stat(path).st_file_attributes & FILE_ATTRIBUTE_HIDDEN) or path.stem.startswith('.')
        except (OSError, AttributeError):
            return path.stem.startswith('.')

    # Displaying the Tree Path [Directories-Nodes]
    def displayPath(self) -> str:
        """
        Method to Display the Path of the Tree [Directories-Nodes]
        :return: String Representation of the Path
        """

        # Check for Parent Directory Name
        if self.parent is None:
            return self.displayName

        # Checking for File-Name Prefix in Tree
        filenamePrefix: str = (
            DirectoryPath.displayNodePrefixLast if self.isLast else DirectoryPath.displayNodePrefixMiddle
        )

        # Adding Prefixes to Beautify Output [List]
        parts: List[str] = [f'{filenamePrefix} {self.displayName}']

        # Adding Prefixes up for Parent-Node Directories
        parent: DirectoryPath = self.parent
        while parent and parent.parent is not None:
            parts.append(
                DirectoryPath.displayParentPrefixMiddle if parent.isLast else DirectoryPath.displayParentPrefixLast
            )
            parent: Path = parent.parent

        return ''.join(reversed(parts))
    
#======================================================================================
# tree_driver.py
#======================================================================================

# Class for Displaying Directory Tree
class DisplayTree:

    # Constructor
    def __init__(
            self,
            dirPath: str='',
            stringRep: bool=False,
            header: bool=False,
            maxDepth: float=float('inf'),
            showHidden: bool=False,
            ignoreList: List[str]=None,
            onlyFiles: bool=False,
            onlyDirs: bool=False,
            sortBy: int=0,
            raiseException: bool=False,
            printErrorTraceback: bool=False
    ) -> None:
        """
        :param dirPath: Root Path of Operation. By Default, Refers to the Current Working Directory
        :param stringRep: Boolean Flag for Direct Console Output or a String Return of the Same. By Default, It Gives out Console Output
        :param header: Boolean Flag for Displaying [OS & Directory Path] Info in the Console. Not Applicable if `string_rep=True`
        :param maxDepth: Max Depth of the Directory Tree. By Default, It goes upto the Deepest Directory/File
        :param showHidden: Boolean Flag for Returning/Displaying Hidden Files/Directories if Value Set to `True`
        :param ignoreList: List of File and Directory Names or Patterns to Ignore
        :param onlyFiles: Boolean Flag to Display Only Files
        :param onlyDirs: Boolean Flag to Display Only Directories
        :param sortBy: Sorting order. Possible Options: [0 - Default, 1 - Files First, 2 - Directories First]
        :param raiseException: Boolean Flag to Raise Exception. By Default, It Doesn't Raise Exception
        :param printErrorTraceback: Boolean Flag to Print Error Traceback. By Default, It Doesn't Print Error Traceback
        :return: None if `string_rep=False` else (str)ing Representation of the Tree
        """

        # Instance Variables
        self.dirPath: str = dirPath
        self.stringRep: bool = stringRep
        self.header: bool = header
        self.maxDepth: float = maxDepth
        self.showHidden: bool = showHidden
        self.ignoreList: List[str] = ignoreList
        self.onlyFiles: bool = onlyFiles
        self.onlyDirs: bool = onlyDirs
        self.sortBy: int = sortBy
        self.raiseException: bool = raiseException
        self.printErrorTraceback: bool = printErrorTraceback

    # Destructor
    def __del__(self):

        del self.dirPath, self.stringRep, self.header, self.maxDepth, self.showHidden, self.ignoreList, \
            self.onlyFiles, self.onlyDirs, self.sortBy, self.raiseException, self.printErrorTraceback

    # Display Function to Print Directory Tree
    @classmethod
    def display(cls, *args: List[Any], **kwargs: dict[str, Any]) -> Union[str, None]:

        # Instance Creation and Display Tree
        instance: DisplayTree = cls(
            *args, **kwargs
        )

        return instance.displayTree()

    # `Display Tree` Method to Print Directory Tree
    def displayTree(self) -> Union[str, None]:

        try:

            # Check for Default Argument
            if self.dirPath:
                self.dirPath: Path = Path(self.dirPath)
            else:
                self.dirPath: Path = Path(getcwd())

            # Build Directory Tree
            paths: str = DirectoryPath.buildTree(
                root=self.dirPath,
                maxDepth=self.maxDepth,
                showHidden=self.showHidden,
                ignoreList=self.ignoreList,
                onlyFiles=self.onlyFiles,
                onlyDirectories=self.onlyDirs,
                sortBy=self.sortBy
            )

            # Check for String Representation
            if self.stringRep:

                # String Representation
                stringOutput: str = str()
                for path in paths:
                    stringOutput += path.displayPath() + '\n'
                return stringOutput

            else:
                # Just Console Print
                if self.header:
                    print(f'''
$ Operating System : {system()}
$ Path : {Path(self.dirPath)}

{'*' * 15} Directory Tree {'*' * 15}
''')

                for path in paths:
                    print(path.displayPath())

        except Exception as expMessage:

            # Exception Handling
            if self.printErrorTraceback:
                print(f'Traceback Details:: {format_exc()}')
            errorMsg: str = f'Exception Occurred! Failed to Generate Tree:: {type(expMessage).__name__}: {expMessage}'
            if self.raiseException:
                raise type(expMessage)(errorMsg)
            else:
                print(errorMsg)
                
#======================================================================================
# This is the end of directory_tree Python module
#======================================================================================