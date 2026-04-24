#======================================================================================
#
# Enhanced Directory Tree with Descriptions
#
# Extension to the directory_tree module adding:
#   - Extension-based file exclusion
#   - Short descriptions for directories (and optionally files)
#   - Fluent helper function for adding descriptions
#
#======================================================================================
# Imports
#======================================================================================

from __future__ import annotations
from typing import Dict, List, Union, Any, Optional
from pathlib import Path
from os import getcwd
from platform import system
from traceback import format_exc

# Import from the provided directory_tree module
from directory_tree import DirectoryPath

#======================================================================================
# Main Class
#======================================================================================

class DisplayTreeWithDescriptions:
    """
    Enhanced directory tree display with descriptions support.
    
    Extends the basic directory tree functionality to include:
    - Short descriptions for directories (and optionally files)
    - Extension-based file exclusion
    - Fluent API for adding descriptions
    - Both console output and string return options
    
    Examples
    --------
    Simple usage (similar to original):
        DisplayTreeWithDescriptions().display('./')
    
    With descriptions:
        tree = DisplayTreeWithDescriptions(
            dirPath='./myproject',
            descriptions={
                'src': 'Source code',
                'tests': 'Test suite',
                '.': 'My awesome project'
            }
        )
        tree.display()
    
    With extension exclusion:
        DisplayTreeWithDescriptions(
            dirPath='./',
            ignoreExtensions=['.pyc', '.log', '.tmp']
        ).display()
    
    Fluent API:
        tree = DisplayTreeWithDescriptions(dirPath='./myproject')
        tree.add_description('src', 'Source code')
        tree.add_description('tests', 'Test suite')
        tree.display()
    """
    
    def __init__(
        self,
        dirPath: str = '',
        stringRep: bool = False,
        header: bool = False,
        maxDepth: float = float('inf'),
        showHidden: bool = False,
        ignoreList: List[str] = None,
        ignoreExtensions: List[str] = None,
        onlyFiles: bool = False,
        onlyDirs: bool = False,
        sortBy: int = 0,
        raiseException: bool = False,
        printErrorTraceback: bool = False,
        descriptions: Dict[str, str] = None,
        descriptionSeparator: str = ' — ',
        showFileDescriptions: bool = False,
    ) -> None:
        """
        Initialize DisplayTreeWithDescriptions.
        
        Parameters
        ----------
        dirPath : str
            Root path of operation. Defaults to current working directory.
        stringRep : bool
            If True, return string instead of printing. Default False.
        header : bool
            If True, show OS and path info header. Default False.
        maxDepth : float
            Maximum depth of tree. Default infinity.
        showHidden : bool
            If True, show hidden files/directories. Default False.
        ignoreList : List[str]
            List of file/directory names or patterns to ignore (fnmatch style).
            Examples: ['*.pyc', 'temp', '__pycache__']
        ignoreExtensions : List[str]
            List of file extensions to ignore. Examples: ['.pyc', '.log', '.tmp']
        onlyFiles : bool
            If True, show only files. Default False.
        onlyDirs : bool
            If True, show only directories. Default False.
        sortBy : int
            Sorting order: 0=Default (alphabetical), 1=Files first, 2=Directories first.
        raiseException : bool
            If True, raise exceptions instead of printing. Default False.
        printErrorTraceback : bool
            If True, print full traceback on error. Default False.
        descriptions : Dict[str, str]
            Dictionary mapping names/paths to descriptions.
            Keys can be: directory name ('src'), file name ('README.md'),
            or relative path ('src/utils', 'tests/test_main.py').
            Use '.' for the root directory.
        descriptionSeparator : str
            Separator between name and description. Default ' — '.
        showFileDescriptions : bool
            If True, show descriptions for files too. Default False (dirs only).
        """
        self.dirPath = dirPath
        self.stringRep = stringRep
        self.header = header
        self.maxDepth = maxDepth
        self.showHidden = showHidden
        self.ignoreList = ignoreList if ignoreList is not None else []
        self.ignoreExtensions = ignoreExtensions if ignoreExtensions is not None else []
        self.onlyFiles = onlyFiles
        self.onlyDirs = onlyDirs
        self.sortBy = sortBy
        self.raiseException = raiseException
        self.printErrorTraceback = printErrorTraceback
        self.descriptions = descriptions if descriptions is not None else {}
        self.descriptionSeparator = descriptionSeparator
        self.showFileDescriptions = showFileDescriptions
        
        self._combinedIgnoreList = self._buildCombinedIgnoreList()
    
    def _buildCombinedIgnoreList(self) -> List[str]:
        """Build combined ignore list including extension patterns."""
        combined = list(self.ignoreList)
        for ext in self.ignoreExtensions:
            if not ext.startswith('*'):
                combined.append('*' + ext)
            else:
                combined.append(ext)
        return combined
    
    def add_description(self, name: str, description: str) -> 'DisplayTreeWithDescriptions':
        """
        Add a description for a directory or file.
        
        Parameters
        ----------
        name : str
            Name of the directory/file, or relative path. Use '.' for root.
        description : str
            Short description text.
            
        Returns
        -------
        DisplayTreeWithDescriptions
            Self, for method chaining.
        """
        self.descriptions[name] = description
        return self
    
    def add_descriptions(self, descriptions: Dict[str, str]) -> 'DisplayTreeWithDescriptions':
        """
        Add multiple descriptions at once.
        
        Parameters
        ----------
        descriptions : Dict[str, str]
            Dictionary of name/path -> description mappings.
            
        Returns
        -------
        DisplayTreeWithDescriptions
            Self, for method chaining.
        """
        self.descriptions.update(descriptions)
        return self
    
    @classmethod
    def display(cls, *args: List[Any], **kwargs: dict[str, Any]) -> Union[str, None]:
        """
        Class method for quick one-line display.
        
        Usage:
            DisplayTreeWithDescriptions.display('./', descriptions={'src': 'Source'})
            
        Returns
        -------
        str or None
            String if stringRep=True, else None (prints to console).
        """
        instance = cls(*args, **kwargs)
        return instance.displayTree()
    
    def displayTree(self) -> Union[str, None]:
        """
        Display the directory tree with descriptions.
        
        Returns
        -------
        str or None
            String representation if stringRep=True, else None (prints to console).
        """
        try:
            if self.dirPath:
                rootPath = Path(self.dirPath)
            else:
                rootPath = Path(getcwd())
            
            paths = DirectoryPath.buildTree(
                root=rootPath,
                maxDepth=self.maxDepth,
                showHidden=self.showHidden,
                ignoreList=self._combinedIgnoreList,
                onlyFiles=self.onlyFiles,
                onlyDirectories=self.onlyDirs,
                sortBy=self.sortBy
            )
            
            if self.stringRep:
                lines = [self._formatLine(p, rootPath) for p in paths]
                return '\n'.join(lines)
            else:
                if self.header:
                    print(f'\n$ Operating System : {system()}')
                    print(f'$ Path : {rootPath}')
                    print(f'\n{"*" * 15} Directory Tree {"*" * 15}\n')
                
                for p in paths:
                    print(self._formatLine(p, rootPath))
                    
        except Exception as expMessage:
            if self.printErrorTraceback:
                print(f'Traceback Details:: {format_exc()}')
            errorMsg = f'Exception Occurred! Failed to Generate Tree:: {type(expMessage).__name__}: {expMessage}'
            if self.raiseException:
                raise type(expMessage)(errorMsg)
            else:
                print(errorMsg)
    
    def _getRelativePath(self, path: Path, rootPath: Path) -> str:
        """Get relative path as string."""
        try:
            rel = path.relative_to(rootPath)
            return str(rel) if str(rel) != '.' else '.'
        except ValueError:
            return path.name
    
    def _formatLine(self, dirPath: DirectoryPath, rootPath: Path) -> str:
        """
        Format a single line with optional description.
        
        Parameters
        ----------
        dirPath : DirectoryPath
            The directory path object from the tree.
        rootPath : Path
            The root path of the tree.
            
        Returns
        -------
        str
            Formatted line with optional description.
        """
        basePath = dirPath.displayPath()
        relativePath = self._getRelativePath(dirPath.path, rootPath)
        name = dirPath.path.name
        
        description = self.descriptions.get(relativePath) or self.descriptions.get(name)
        
        if description:
            if dirPath.path.is_dir() or self.showFileDescriptions:
                basePath += self.descriptionSeparator + description
        
        return basePath
    
    def get_tree_structure(self) -> List[str]:
        """
        Get list of directory/file paths in the tree.
        
        Useful for reference when adding descriptions - shows you what
        paths are available to describe.
        
        Returns
        -------
        List[str]
            List of relative paths in the tree.
        """
        try:
            if self.dirPath:
                rootPath = Path(self.dirPath)
            else:
                rootPath = Path(getcwd())
            
            paths = DirectoryPath.buildTree(
                root=rootPath,
                maxDepth=self.maxDepth,
                showHidden=self.showHidden,
                ignoreList=self._combinedIgnoreList,
                onlyFiles=self.onlyFiles,
                onlyDirectories=self.onlyDirs,
                sortBy=self.sortBy
            )
            
            return [self._getRelativePath(p.path, rootPath) for p in paths]
        except Exception:
            return []


#======================================================================================
# Helper Class
#======================================================================================

class TreeDescriptor:
    """
    Helper class for interactively describing a directory tree.
    
    Provides a fluent API for adding descriptions to a directory tree
    before displaying it. Use the describe_tree() function to create instances.
    
    Examples
    --------
    Basic usage with callable syntax:
        desc = describe_tree('./myproject')
        desc('src', 'Source code')
        desc('tests', 'Test suite')
        desc.display()
    
    Chained callable syntax:
        describe_tree('./myproject')('src', 'Source code')('tests', 'Test suite').display()
    
    Method syntax:
        describe_tree('./myproject').desc('src', 'Source code').desc('tests', 'Test suite').display()
    
    With root description:
        describe_tree('./myproject').root_desc('My Project').desc('src', 'Source').display()
    
    Add multiple at once:
        describe_tree('./myproject').add_many({'src': 'Source', 'tests': 'Tests'}).display()
    """
    
    def __init__(
        self,
        path: str,
        ignoreList: Optional[List[str]],
        ignoreExtensions: Optional[List[str]],
        showFiles: bool,
        showHidden: bool,
        maxDepth: float,
        sortBy: int,
        descriptionSeparator: str,
        showFileDescriptions: bool
    ):
        self._path = path
        self._ignoreList = ignoreList
        self._ignoreExtensions = ignoreExtensions
        self._showFiles = showFiles
        self._showHidden = showHidden
        self._maxDepth = maxDepth
        self._sortBy = sortBy
        self._descriptionSeparator = descriptionSeparator
        self._showFileDescriptions = showFileDescriptions
        self._descriptions: Dict[str, str] = {}
    
    def __call__(self, name: str, description: str) -> 'TreeDescriptor':
        """
        Add a description using callable syntax.
        
        Parameters
        ----------
        name : str
            Directory/file name or relative path. Use '.' for root.
        description : str
            Short description text.
            
        Returns
        -------
        TreeDescriptor
            Self, for chaining.
        """
        self._descriptions[name] = description
        return self
    
    def desc(self, name: str, description: str) -> 'TreeDescriptor':
        """
        Add a description using method syntax.
        
        Parameters
        ----------
        name : str
            Directory/file name or relative path.
        description : str
            Short description text.
            
        Returns
        -------
        TreeDescriptor
            Self, for chaining.
        """
        self._descriptions[name] = description
        return self
    
    def root_desc(self, description: str) -> 'TreeDescriptor':
        """
        Set description for the root directory.
        
        Parameters
        ----------
        description : str
            Short description for the root directory.
            
        Returns
        -------
        TreeDescriptor
            Self, for chaining.
        """
        self._descriptions['.'] = description
        return self
    
    def add_many(self, descriptions: Dict[str, str]) -> 'TreeDescriptor':
        """
        Add multiple descriptions at once.
        
        Parameters
        ----------
        descriptions : Dict[str, str]
            Dictionary of name/path -> description mappings.
            
        Returns
        -------
        TreeDescriptor
            Self, for chaining.
        """
        self._descriptions.update(descriptions)
        return self
    
    def get_structure(self) -> List[str]:
        """
        Get list of paths in the tree (useful for knowing what to describe).
        
        Returns
        -------
        List[str]
            List of relative paths in the tree.
        """
        tree = DisplayTreeWithDescriptions(
            dirPath=self._path,
            ignoreList=self._ignoreList,
            ignoreExtensions=self._ignoreExtensions,
            onlyDirs=not self._showFiles,
            showHidden=self._showHidden,
            maxDepth=self._maxDepth,
            sortBy=self._sortBy
        )
        return tree.get_tree_structure()
    
    def display(self, header: bool = False) -> None:
        """
        Display the tree with descriptions to console.
        
        Parameters
        ----------
        header : bool
            If True, show OS and path info header.
        """
        DisplayTreeWithDescriptions(
            dirPath=self._path,
            ignoreList=self._ignoreList,
            ignoreExtensions=self._ignoreExtensions,
            onlyDirs=not self._showFiles,
            showHidden=self._showHidden,
            maxDepth=self._maxDepth,
            sortBy=self._sortBy,
            descriptions=self._descriptions,
            descriptionSeparator=self._descriptionSeparator,
            showFileDescriptions=self._showFileDescriptions,
            header=header
        ).displayTree()
    
    def to_string(self) -> str:
        """
        Get the tree as a string.
        
        Returns
        -------
        str
            String representation of the tree with descriptions.
        """
        result = DisplayTreeWithDescriptions(
            dirPath=self._path,
            ignoreList=self._ignoreList,
            ignoreExtensions=self._ignoreExtensions,
            onlyDirs=not self._showFiles,
            showHidden=self._showHidden,
            maxDepth=self._maxDepth,
            sortBy=self._sortBy,
            descriptions=self._descriptions,
            descriptionSeparator=self._descriptionSeparator,
            showFileDescriptions=self._showFileDescriptions,
            stringRep=True
        ).displayTree()
        return result if result else ''
    
    def __str__(self) -> str:
        """String representation of the tree with descriptions."""
        return self.to_string()
    
    def __repr__(self) -> str:
        """Developer representation."""
        desc_preview = dict(list(self._descriptions.items())[:3])
        if len(self._descriptions) > 3:
            desc_preview['...'] = f'({len(self._descriptions) - 3} more)'
        return f"TreeDescriptor(path='{self._path}', descriptions={desc_preview})"


#======================================================================================
# Helper Function
#======================================================================================

def describe_tree(
    path: str = './',
    ignoreList: List[str] = None,
    ignoreExtensions: List[str] = None,
    showFiles: bool = False,
    showHidden: bool = False,
    maxDepth: float = float('inf'),
    sortBy: int = 2,
    descriptionSeparator: str = ' — ',
    showFileDescriptions: bool = False
) -> TreeDescriptor:
    """
    Helper function to create a tree descriptor for adding descriptions.
    
    Returns a TreeDescriptor object that allows you to add descriptions
    in a fluent, chainable manner before displaying the tree.
    
    Parameters
    ----------
    path : str
        Root path of the tree. Default './'.
    ignoreList : List[str]
        List of file/directory patterns to ignore (fnmatch style).
        Examples: ['*.pyc', '__pycache__', 'temp*']
    ignoreExtensions : List[str]
        List of file extensions to ignore.
        Examples: ['.pyc', '.log', '.tmp', '.o']
    showFiles : bool
        If True, include files in the tree. Default False (dirs only).
    showHidden : bool
        If True, show hidden files/directories. Default False.
    maxDepth : float
        Maximum depth of the tree. Default infinity.
    sortBy : int
        Sorting order: 0=Alphabetical, 1=Files first, 2=Directories first.
        Default is 2 (directories first, which usually looks cleaner).
    descriptionSeparator : str
        Separator between name and description. Default ' — '.
    showFileDescriptions : bool
        If True, allow descriptions for files (requires showFiles=True).
        Default False (only directory descriptions).
    
    Returns
    -------
    TreeDescriptor
        A descriptor object for adding descriptions and displaying.
    
    Examples
    --------
    Basic usage with callable syntax:
    
        desc = describe_tree('./myproject')
        desc('src', 'Source code directory')
        desc('tests', 'Test suite')
        desc('docs', 'Documentation')
        desc.display()
    
    Chained callable syntax (one-liner):
    
        describe_tree('./myproject')('src', 'Source code')('tests', 'Test suite').display()
    
    With root description using method syntax:
    
        describe_tree('./myproject').root_desc('My Awesome Project v1.0').desc('src', 'Source code').desc('tests', 'Test suite').display(header=True)
    
    With files and extension filtering:
    
        desc = describe_tree('./myproject', showFiles=True, ignoreExtensions=['.pyc'])
        desc('src', 'Source code')
        desc('src/main.py', 'Main entry point')
        desc('README.md', 'Project documentation')
        desc.display()
    
    Get as string:
    
        desc = describe_tree('./myproject')
        desc('src', 'Source code')
        tree_str = desc.to_string()
        print(tree_str)
    
    See available paths to describe:
    
        desc = describe_tree('./myproject')
        print(desc.get_structure())  # Output: ['.', 'src', 'tests', 'docs', ...]
    """
    return TreeDescriptor(
        path=path,
        ignoreList=ignoreList,
        ignoreExtensions=ignoreExtensions,
        showFiles=showFiles,
        showHidden=showHidden,
        maxDepth=maxDepth,
        sortBy=sortBy,
        descriptionSeparator=descriptionSeparator,
        showFileDescriptions=showFileDescriptions
    )


#======================================================================================
# Quick Reference Examples (run as main to see demos)
#======================================================================================

if __name__ == '__main__':
    
    print("=" * 60)
    print("EXAMPLE 1: Basic usage (like original DisplayTree)")
    print("=" * 60)
    DisplayTreeWithDescriptions().display('./')
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: With descriptions (dict-based)")
    print("=" * 60)
    DisplayTreeWithDescriptions.display(
        './',
        maxDepth=2,
        descriptions={
            '.': 'Project Root',
            'directory_tree': 'Original module',
        }
    )
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Using describe_tree() helper")
    print("=" * 60)
    desc = describe_tree('./', maxDepth=2)
    desc.root_desc('Project Root')
    desc('directory_tree', 'Original module')
    desc.display()
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: With files and extension exclusion")
    print("=" * 60)
    desc = describe_tree('./', showFiles=True, ignoreExtensions=['.pyc'], maxDepth=1)
    desc.root_desc('Project Root')
    desc.display()
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Get as string")
    print("=" * 60)
    desc = describe_tree('./', maxDepth=1)
    desc.root_desc('Project Root')
    tree_str = desc.to_string()
    print(f"String output:\n{tree_str}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 6: See available paths to describe")
    print("=" * 60)
    desc = describe_tree('./', maxDepth=2)
    print("Available paths:", desc.get_structure())