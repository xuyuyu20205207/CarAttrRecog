<template>
  <a-layout id="components-layout-demo-custom-trigger">
    <a-layout-sider v-model="collapsed" :trigger="null" collapsible>
      <div class="logo" />
      <a-menu theme="dark" mode="inline" :default-selected-keys="['1']">
        <a-menu-item key="1">
          <user-outlined />
          <span>优图速算</span>
          <router-link :to="{ name: 'cv', params: {my_key : 1}}"></router-link>
        </a-menu-item>
        <a-menu-item key="2">
          <video-camera-outlined />
          <span>车辆识别</span>
          <router-link :to="{ name: 'cv', params: {my_key : 2}}"></router-link>
        </a-menu-item>
      </a-menu>
    </a-layout-sider>
    <a-layout>
      <a-layout-header style="background: #fff; padding: 0">
        <menu-unfold-outlined
          class="trigger"
          v-if="collapsed"
          @click="() => (collapsed = !collapsed)"
        />
        <menu-fold-outlined v-else class="trigger" @click="() => (collapsed = !collapsed)" />
      </a-layout-header>
      <a-page-header
        style="border: 1px solid rgb(235, 237, 240)"
        title="AI识别系统 "
        sub-title="请上传图片"
      />
      <a-layout-content
        :style="{
          margin: '4px 16px',
          padding: '24px',
          background: '#fff',
          minHeight: '580px',
        }"
      >
      <router-view name="main" :key="$route.fullPath"></router-view>
      </a-layout-content>
    </a-layout>
  </a-layout>
</template>

<script lang="ts">
import {
  UserOutlined,
  VideoCameraOutlined,
  UploadOutlined,
  MenuUnfoldOutlined,
  MenuFoldOutlined,
} from '@ant-design/icons-vue';
import { defineComponent, ref } from 'vue';
export default defineComponent({
  components: {
    UserOutlined,
    VideoCameraOutlined,
    UploadOutlined,
    MenuUnfoldOutlined,
    MenuFoldOutlined,
  },
  setup() {
    return {
      selectedKeys: ref<string[]>(['1']),
      collapsed: ref<boolean>(false),
    };
  },
});
</script>

<script>
import ImageUpload from './components/ImageUpload.vue';
//import axios from "axios";
export default {
  components: { ImageUpload },
}
</script>
<style>
.a-upload {
  width: 600px;
  height: 200px;
  display: inline-block;
  vertical-align: top;
}
.result-img {
  width: 500px;
  height: 200px;
  display: inline-block;
  vertical-align: top;
  position: absoulte;
}
.ant-upload-select-picture-card i {
  font-size: 32px;
  color: #999;
}

.ant-upload-select-picture-card .ant-upload-text {
  margin-top: 8px;
  color: #666;
}

#components-layout-demo-custom-trigger .trigger {
  font-size: 18px;
  line-height: 64px;
  padding: 0 24px;
  cursor: pointer;
  transition: color 0.3s;
}

#components-layout-demo-custom-trigger .trigger:hover {
  color: #1890ff;
}

#components-layout-demo-custom-trigger .logo {
  height: 32px;
  background: rgba(255, 255, 255, 0.3);
  margin: 16px;
}

.site-layout .site-layout-background {
  background: #fff;
}
</style>
