#!/bin/bash
# PaddleOCR-VL-API Docker 镜像构建脚本（交互式）

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 默认配置
REGISTRY="quantatrisk"
IMAGE_NAME="paddleocr-api"
VERSION="latest"
PLATFORM="linux/amd64"
PUSH="false"
EXPORT_TAR="false"
EXPORT_DIR="."
INTERACTIVE="true"
NO_CACHE="false"

# 打印带颜色的消息
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header() { echo -e "\n${BOLD}${BLUE}$1${NC}\n"; }

# 显示 banner
show_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
  ____           _     _ _       ___   ____ ____     __     ___     _    ____ ___
 |  _ \ __ _  __| | __| | | ___ / _ \ / ___|  _ \    \ \   / / |   / \  |  _ \_ _|
 | |_) / _` |/ _` |/ _` | |/ _ \ | | | |   | |_) |____\ \ / /| |  / _ \ | |_) | |
 |  __/ (_| | (_| | (_| | |  __/ |_| | |___|  _ <______\ V / | |_/ ___ \|  __/| |
 |_|   \__,_|\__,_|\__,_|_|\___|\___/ \____|_| \_\      \_/  |_(_)_/   \_\_|  |___|

              Docker 镜像构建工具 v1.0 (GPU Only)
EOF
    echo -e "${NC}"
}

# 显示帮助
show_help() {
    cat << EOF
PaddleOCR-VL-API Docker 镜像构建脚本 (GPU 版本)

用法: ./build.sh [选项]

选项:
    -v, --version VER     版本标签 (默认: latest)
    -p, --push            构建后推送到 Docker Hub
    -e, --export          导出为 tar.gz 文件
    -o, --output DIR      导出目录 (默认: 当前目录)
    -r, --registry REG    镜像仓库 (默认: quantatrisk)
    -n, --no-cache        不使用缓存，强制重新安装依赖
    -y, --yes             跳过交互确认
    -h, --help            显示帮助

示例:
    ./build.sh                      # 交互式构建
    ./build.sh -v v1.0.0            # 构建指定版本
    ./build.sh -v v1.0.0 -p         # 构建并推送
    ./build.sh -v latest -e         # 构建并导出为 tar.gz
    ./build.sh -n                   # 不使用缓存构建
    ./build.sh -y -p                # 跳过确认并推送

注意:
    本项目仅支持 GPU 版本，需要 NVIDIA GPU 运行环境
    镜像基于 CUDA 12.1，仅支持 AMD64 (x86_64) 架构

    模型文件不包含在镜像中，需要通过 Volume 挂载:
    详见 docker-compose.yml 中的 volumes 配置

构建的镜像标签:
    ${REGISTRY}/${IMAGE_NAME}:latest
    ${REGISTRY}/${IMAGE_NAME}:<version>

EOF
}

# 简单选择（数字输入）
simple_select() {
    local prompt="$1"
    shift
    local options=("$@")

    echo -e "\n${BOLD}${prompt}${NC}" >&2
    for i in "${!options[@]}"; do
        echo -e "  ${GREEN}$((i+1))${NC}) ${options[$i]}" >&2
    done

    while true; do
        echo -ne "\n请输入选项 [1-${#options[@]}]: " >&2
        read -r choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]; then
            echo $((choice - 1))
            return
        fi
        echo -e "${RED}无效选项，请重新输入${NC}" >&2
    done
}

# 交互式配置
interactive_config() {
    show_banner

    # 架构说明
    header "步骤 1/5: 架构说明"
    echo -e "${CYAN}本项目仅支持 GPU 版本${NC}"
    echo -e "  目标架构: ${GREEN}linux/amd64 (x86_64)${NC}"
    echo -e "  CUDA 版本: ${GREEN}12.1${NC}"
    echo -e "  要求: NVIDIA GPU + nvidia-docker2"

    # 输入版本号
    header "步骤 2/5: 设置版本标签"
    echo -ne "请输入版本标签 [默认: latest]: "
    read -r input_version
    [ -n "$input_version" ] && VERSION="$input_version"

    # 是否推送
    header "步骤 3/5: 推送设置"
    local push_opts=("仅本地构建 (不推送)" "构建并推送到 Docker Hub")
    local push_idx=$(simple_select "选择推送选项:" "${push_opts[@]}")
    [ $push_idx -eq 1 ] && PUSH="true"

    # 是否导出为 tar.gz
    header "步骤 4/5: 导出设置"
    local export_opts=("不导出" "导出为 tar.gz 文件")
    local export_idx=$(simple_select "是否导出镜像为 tar.gz:" "${export_opts[@]}")
    if [ $export_idx -eq 1 ]; then
        EXPORT_TAR="true"
        echo -ne "请输入导出目录 [默认: 当前目录]: "
        read -r input_dir
        [ -n "$input_dir" ] && EXPORT_DIR="$input_dir"
    fi

    # 是否使用缓存
    header "步骤 5/5: 构建缓存"
    local cache_opts=("使用缓存 (更快)" "不使用缓存 (强制重新安装依赖)")
    local cache_idx=$(simple_select "选择构建缓存策略:" "${cache_opts[@]}")
    [ $cache_idx -eq 1 ] && NO_CACHE="true"

    # 确认配置
    header "配置确认"
    echo -e "  镜像名称:   ${CYAN}${REGISTRY}/${IMAGE_NAME}:${VERSION}${NC}"
    echo -e "  目标架构:   ${CYAN}${PLATFORM}${NC}"
    echo -e "  版本标签:   ${CYAN}${VERSION}${NC}"
    echo -e "  镜像仓库:   ${CYAN}${REGISTRY}${NC}"
    echo -e "  推送镜像:   ${CYAN}$([ "$PUSH" = "true" ] && echo "是" || echo "否")${NC}"
    echo -e "  导出tar.gz: ${CYAN}$([ "$EXPORT_TAR" = "true" ] && echo "是 → ${EXPORT_DIR}" || echo "否")${NC}"
    echo -e "  使用缓存:   ${CYAN}$([ "$NO_CACHE" = "true" ] && echo "否 (强制重新安装)" || echo "是")${NC}"

    echo ""
    echo -ne "确认开始构建? [Y/n]: "
    read -r confirm
    if [[ "$confirm" =~ ^[Nn] ]]; then
        echo -e "${YELLOW}已取消构建${NC}"
        exit 0
    fi
}

# 检查 buildx
check_buildx() {
    if ! docker buildx version &> /dev/null; then
        error "需要 Docker Buildx 支持，请先安装"
    fi

    # 检查/创建 builder
    if ! docker buildx inspect paddleocr-builder &> /dev/null; then
        info "创建 buildx builder..."
        docker buildx create --name paddleocr-builder --use
    else
        docker buildx use paddleocr-builder
    fi
}

# 构建镜像
build_image() {
    local tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

    info "构建 PaddleOCR-VL-API 镜像: $tag"
    info "目标架构: $PLATFORM"

    local build_args="--platform $PLATFORM -t $tag -f Dockerfile"

    # 添加 latest 标签（如果版本不是 latest）
    if [ "$VERSION" != "latest" ]; then
        build_args="$build_args -t ${REGISTRY}/${IMAGE_NAME}:latest"
    fi

    # 添加 --no-cache 参数
    if [ "$NO_CACHE" = "true" ]; then
        build_args="$build_args --no-cache"
        info "已启用 --no-cache，将强制重新安装所有依赖"
    fi

    # 推送或加载
    if [ "$PUSH" = "true" ]; then
        build_args="$build_args --push"
    elif [ "$EXPORT_TAR" = "true" ]; then
        # 直接通过 buildx 导出为 tar
        local tar_file="${EXPORT_DIR}/${IMAGE_NAME}-${VERSION}-amd64.tar"
        mkdir -p "$EXPORT_DIR"
        build_args="$build_args --output type=docker,dest=${tar_file}"
        info "将直接导出到: ${tar_file}.gz"
    else
        build_args="$build_args --load"
    fi

    docker buildx build $build_args .

    info "镜像构建完成: $tag"

    # 如果使用了 --output 导出，压缩 tar 文件
    if [ "$EXPORT_TAR" = "true" ] && [ "$PUSH" != "true" ]; then
        local tar_file="${EXPORT_DIR}/${IMAGE_NAME}-${VERSION}-amd64.tar"
        if [ -f "$tar_file" ]; then
            info "压缩: ${tar_file} → ${tar_file}.gz"
            gzip -f "$tar_file"
            local size=$(du -h "${tar_file}.gz" | cut -f1)
            info "导出成功: ${tar_file}.gz ($size)"
        fi
    fi
}

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -p|--push)
                PUSH="true"
                shift
                ;;
            -e|--export)
                EXPORT_TAR="true"
                shift
                ;;
            -o|--output)
                EXPORT_DIR="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -n|--no-cache)
                NO_CACHE="true"
                shift
                ;;
            -y|--yes)
                INTERACTIVE="false"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "未知选项: $1"
                ;;
        esac
    done
}

# 主流程
main() {
    parse_args "$@"

    # 如果没有通过参数指定版本，进入交互模式
    if [ "$INTERACTIVE" = "true" ]; then
        interactive_config
    fi

    # 检查 buildx
    check_buildx

    header "开始构建"
    echo -e "  镜像名称:   ${CYAN}${REGISTRY}/${IMAGE_NAME}:${VERSION}${NC}"
    echo -e "  目标架构:   ${CYAN}${PLATFORM}${NC}"
    echo -e "  版本标签:   ${CYAN}${VERSION}${NC}"
    echo -e "  推送镜像:   ${CYAN}$([ "$PUSH" = "true" ] && echo "是" || echo "否")${NC}"
    echo -e "  导出tar.gz: ${CYAN}$([ "$EXPORT_TAR" = "true" ] && echo "是 → ${EXPORT_DIR}" || echo "否")${NC}"
    echo -e "  使用缓存:   ${CYAN}$([ "$NO_CACHE" = "true" ] && echo "否 (强制重新安装)" || echo "是")${NC}"
    echo -e "  Dockerfile: ${CYAN}Dockerfile${NC}"
    echo ""

    # 执行构建
    build_image

    header "构建完成!"

    # 显示构建的镜像（仅当加载到本地时）
    if [ "$PUSH" != "true" ] && [ "$EXPORT_TAR" != "true" ]; then
        info "已构建的镜像:"
        docker images | grep "${REGISTRY}/${IMAGE_NAME}" | head -10
    fi

    # 显示导出的文件
    if [ "$EXPORT_TAR" = "true" ]; then
        echo ""
        info "导出的镜像文件:"
        ls -lh "${EXPORT_DIR}"/${IMAGE_NAME}*.tar.gz 2>/dev/null | tail -5
        echo ""
        info "使用以下命令加载镜像:"
        echo "  gunzip -c <file>.tar.gz | docker load"
    fi

    # 显示后续步骤
    if [ "$PUSH" != "true" ]; then
        echo ""
        info "后续步骤:"
        echo "  1. 启动服务: docker-compose up -d"
        echo "  2. 查看日志: docker-compose logs -f"
        echo "  3. 测试 API: curl http://localhost:8781/health"
    fi
}

main "$@"
