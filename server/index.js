const express = require('express');
const app = express();
const mountMiddleware = require('./config/middleware');
const mountRoutes = require('./config/routes');
const port = process.env.PORT || 3000;

require('./config/sequelize')

mountMiddleware(app);
mountRoutes(app);

app.listen(port, () => {
  console.log(`listening on ${port}`);
});

module.exports = app;
