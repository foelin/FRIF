# FRIF: Fast Robust Invariant Feature


FRIF is open source with a [public repository](https://github.com/foelin/FRIF.git) on GitHub.
Current implementation is based on the source code of BRISK provided
by Stefan Leutenegger et al.

You can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
You should have received a copy of the GNU General Public License
along with FRIF.  If not, see <http://www.gnu.org/licenses/>.

### Usage
Command line arguments:

    -img                    the input image file
    -kp 					the input region file 
	-des 					the output descriptor file
	-blobThresh	[40.0]		the threshold of FRIF detector
	-border		[50]		the border of image
	-supLine	[1]			do the line suppression or not
	-lineThresh	[20]		the threshold for line suppression
	
### Version
1.0

### Requirement
OpenCV 3.0

### Reference:
[1] Zhenhua Wang, Bin Fan and Fuchao Wu, FRIF: Fast Robust Invariant Feature, in British Machine Vision Conference, 2013

