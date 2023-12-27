# Dream Content Analysis with Topic Modeling

Dream content analysis has been done manually by psychologists for some time. Many systems
have been developed to quantify and relate the “content” of a given dream, from many different
perspectives; notably, the Hall/Van de Castle System of quantitative dream content analysis.
This system differs from others in being composed from formally defined quantitative accounts of a
handful of qualitative categories. Typical content analysis defines content as being the whole of all
constituent elements; thoughts, emotions, images, and so on; this is largely empirical in nature when
not given the constraints of a formally defined quantitative system, an issue the HVC system avoids.

The pipeline that I have developed provides an interesting algorithmic measure to identify dream content. The use cases of these methods are broad
and have plenty of room for improvement by experimenting with different combinations of semantic
relatedness model, derivations of data for such models, and pruning methods. Other methods of
information extraction may be used, such as sentiment analysis or conjoined use with text analysis
based upon neural networks. As the project stands, it is a great tool for therapeutic use and as an aid in
dream interpretation.

## Setup

This project was only tested on Debian-based Linux. The data folder contains:
 - Images
 - .dot files: visualize these in Gephi
 - .txt files: These are proximity data and term files for use in JPathfinder.

You can find JPathfinder here:
https://research-collective.com/PFWeb/Download.html

Using Python 3.10, the necessary packages are:
sklearn
networkx
numpy
matplotlib
contractions
regex
