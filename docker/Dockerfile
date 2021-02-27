#FROM rustembedded/cross:arm-unknown-linux-gnueabihf
FROM zenria/cross:arm-rpi-4.9.3-linux-gnueabihf

#ENV PKG_CONFIG_PATH="/usr/lib/arm-linux-gnueabihf/pkgconfig"
#ENV RUST_BACKTRACE=full
#ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/arm-linux-gnueabihf"

#RUN dpkg --add-architecture armhf &&\
#		apt-get update &&\
#		apt-get install -y \
#			libasound2-dev:armhf

RUN curl -sL http://files.portaudio.com/archives/pa_stable_v190600_20161030.tgz | tar xzvf - &&\
    cd portaudio &&\
    AR=arm-linux-gnueabihf-ar CC=arm-linux-gnueabihf-gcc ./configure \
			--host=arm-linux-gnueabihf \
			--with-alsa &&\
    make -j4 &&\
    make install

#FROM rustembedded/cross:arm-unknown-linux-gnueabihf

#ENV PKG_CONFIG_PATH="/usr/lib/arm-linux-gnueabihf/pkgconfig"
#ENV RUST_BACKTRACE=full
#ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/arm-linux-gnueabihf:/usr/local/lib"

#RUN dpkg --add-architecture armhf &&\
#		apt-get update &&\
#		apt-get install -y \
#			curl

#RUN curl -sL https://github.com/raspberrypi/tools/archive/5caa7046982f0539cf5380f94da04b31129ed521.tar.gz  | tar xzf - -C /usr/local --strip-components=1

#RUN rm -rf /usr/arm-linux-gnu* 

#RUN mv /usr/local/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf /usr/arm-linux-gnueabihf

#RUN rm -rf /usr/local/arm-bcm2708

#RUN apt-get install -y \
#    libjack-dev:armhf \
#    libdb5.3-dev:armhf \
#    portaudio19-dev:armhf

#RUN curl -sL ftp://ftp.alsa-project.org/pub/lib/alsa-lib-1.2.4.tar.bz2 | tar jxf - &&\
#    cd alsa-lib-1.2.4 &&\
#    AR=arm-linux-gnueabihf-ar CC=arm-linux-gnueabihf-gcc ./configure \
#      --host=arm-linux-gnueabihf \
#      --prefix=/usr/arm-linux-gnueabihf/arm-linux-gnueabihf &&\
#    make -j4 &&\
#    make install

#RUN curl -sL http://files.portaudio.com/archives/pa_stable_v190600_20161030.tgz | tar xzvf - &&\
#    cd portaudio &&\
#    AR=arm-linux-gnueabihf-ar CC=arm-linux-gnueabihf-gcc ./configure \
#			--host=arm-linux-gnueabihf \
#			--with-alsa &&\
#		make -j4 &&\
#		make install
