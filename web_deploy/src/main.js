/*
 * @Description: 
 * @Author: Charles
 * @Date: 2021-06-27 13:48:56
 * @LastEditTime: 2021-06-27 20:06:46
 * @LastEditors: Charles
 * @Reference: 
 */
import { createApp } from 'vue'
import './index.css'
import Antd from 'ant-design-vue';
import App from './App.vue';

import 'ant-design-vue/dist/antd.css';
import ImageUpload from './components/ImageUpload.vue';
import { createRouter, createWebHistory } from "vue-router";

// Vue.config.productionTip = false
// Vue.prototype.$axios = axios
// axios.defaults.baseURL = '/api'      

const app = createApp(App);
app.config.productionTip = false;
const router = createRouter({
    history: createWebHistory(),
    routes: [
      {
        path: "/cv/:my_key",
        name: "cv",
        components: {
          main: ImageUpload,
        },
      },
    ]
});


app.use(Antd);
app.use(router);
app.mount('#app');
