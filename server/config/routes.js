const mountStockRoutes = require('../stocks/stocksRoutes');

const mountRoutes = (app) => {
  mountStockRoutes(app);
};

module.exports = mountRoutes;
