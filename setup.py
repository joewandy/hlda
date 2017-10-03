from distutils.core import setup
setup(
  name = 'hlda',
  packages = ['hlda'], # this must be the same as the name above
  version = '0.2',
  description = 'Gibbs sampler for the Hierarchical Latent Dirichlet Allocation topic model. This is based on the hLDA implementation from Mallet, having a fixed depth on the nCRP tree.',
  author = 'Joe Wandy',
  author_email = 'joewandy@gmail.com',
  url = 'https://github.com/joewandy/hlda', # use the URL to the github repo
  download_url = 'https://github.com/joewandy/hlda/archive/0.2.tar.gz', # I'll explain this in a second
  keywords = ['topic', 'model', 'lda', 'gibbs', 'sampler', 'hlda'], # arbitrary keywords
  classifiers = [],
)