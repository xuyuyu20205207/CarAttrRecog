/*
 * @Description: 
 * @Author: Charles
 * @Date: 2021-06-27 19:51:10
 * @LastEditTime: 2021-06-27 20:09:09
 * @LastEditors: Charles
 * @Reference: 
 */
module.exports = {
    // cli3 代理是从指定的target后面开始匹配的，不是任意位置；配置pathRewrite可以做替换
    devServer: {
      proxy: {
        // 静态图片资源(因为放到了CDN上)
        '/api': {
          target: 'http://127.0.0.1:5000/api/',
          changeOrigin: true,
          pathRewrite: {
            '/api': ''
          }
        },
      }
    }
  }