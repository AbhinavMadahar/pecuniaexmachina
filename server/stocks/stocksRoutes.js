stockCtrl = require('./stocksController')

const mountStockRoutes = app => {
  app.route('/api/stocks')
    .get(stockCtrl.getStock)
};

module.exports = mountStockRoutes;
