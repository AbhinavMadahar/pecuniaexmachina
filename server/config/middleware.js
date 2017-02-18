const bodyParser = require('body-parser');
const morgan = require('morgan');
const path = require('path')

const mountMiddleware = app => {
  app.use(bodyParser.urlencoded({ extended: true }))
  app.use(bodyParser.json());
  app.use(morgan('dev'));
};

module.exports = mountMiddleware;
