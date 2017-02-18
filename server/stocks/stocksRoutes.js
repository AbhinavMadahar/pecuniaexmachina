stockCtrl = require('./stocksController')

const mountStockRoutes = app => {
  app.route('/api/stocks')
    .post(stockCtrl.addStock)
};

module.exports = mountStockRoutes;
